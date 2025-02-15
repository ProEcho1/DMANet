import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention1, AttentionLayer
from layers.RevIN import RevIN
from einops import rearrange
from torch.nn.functional import gumbel_softmax
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_model
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, configs.d_model)
        self.W_pos = nn.Parameter(torch.empty(1, 1, configs.d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Linear(configs.d_model, self.pred_len))
        self.fc = nn.Sequential(
            nn.Linear(configs.d_model, self.d_ff),
            nn.LeakyReLU(),
            nn.Linear(self.d_ff,configs.d_model)
        )
        self.layernorm = nn.LayerNorm(configs.d_model)
        self.layernorm1 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.down_sampling_layers = configs.down_sampling_layers
        if configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(configs.down_sampling_window, return_indices=False)
        elif configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(configs.down_sampling_window)
        elif configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        self.fusion_layer = nn.Sequential(
            nn.Linear(configs.down_sampling_layers * configs.d_model,configs.d_model),
            nn.LeakyReLU()
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayerModule(configs, layer_num=i+1)
            for i in range(self.down_sampling_layers)
        ])
        self.freq_lens = [
            math.ceil((configs.d_model // 2 + 1) / (self.down_sampling_window ** i))
            for i in range(configs.down_sampling_layers)
        ]
        self.freq_layers = nn.ModuleList([
            FrequencyProcessingLayer(
                configs.enc_in, configs.c_out, freq_len,
                configs.down_sampling_window, configs.down_sampling_window
            ) for freq_len in self.freq_lens
        ])
        self.sampling_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                math.ceil((configs.d_model // 2 + 1) / (self.down_sampling_window ** i)),
                        configs.d_model
                    ).to(torch.cfloat)
                for i in (range(configs.down_sampling_layers))
            ])


    def downsample(self, x):

        downsampled_data = [x]
        for _ in range(self.down_sampling_layers - 1):
            x = self.down_pool(x)
            downsampled_data.append(x)
        return downsampled_data

    def encoder(self, x):

        x=x.permute(0, 2, 1)
        x = self.embedding(x) + self.W_pos
        x = self.layernorm(x)
        downsampled_data = self.downsample(x)

        encoded_features = []
        for i, data in enumerate(downsampled_data):
            freq = torch.fft.rfft(data, dim=-1, norm="ortho")
            data = self.freq_layers[i](freq)
            data = self.sampling_layers[i](data)
            encoded_feature = self.encoder_layers[i](data)
            encoded_feature = torch.fft.irfft(encoded_feature, n=self.d_model, dim=-1, norm="ortho")
            encoded_features.append(encoded_feature)

        multi_scale_features = torch.cat(encoded_features, dim=-1)
        global_features = self.fusion_layer(multi_scale_features)

        return global_features

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        z = x_enc
        z = self.revin_layer(z, 'norm')
        x = z

        freq_features = self.encoder(x)

        fused_features = self.layernorm1(freq_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.fc(fused_features)
        output = self.output(fused_features)
        output = output.permute(0, 2, 1)

        z = output
        z = self.revin_layer(z, 'denorm')
        output = z

        return output

class EncoderLayerModule(nn.Module):
    def __init__(self, configs,layer_num):
        super(EncoderLayerModule, self).__init__()
        self.d_model=configs.d_model
        self.scale = 0.02
        self.d_ff = configs.d_model
        self.down_sampling_layers = configs.down_sampling_layers
        self.layer_num = layer_num

        self.global_topk = configs.global_topk
        self.local_band_count = self.down_sampling_layers - self.layer_num +1 #configs.local_band_count
        self.local_topk = self.calculate_local_topk()
        self.all_topk = self.global_topk + self.local_band_count * self.local_topk

        self.Mahalanobis_mask = Mahalanobis_mask(self.all_topk)
        self.freq_attention = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention1(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    self.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x):
        B, C, L = x.size()
        freq_magnitude = torch.abs(x)
        freq_phase = torch.angle(x)

        key_freq_indices = self.cumulative_energy_dynamic(
            freq_magnitude, self.global_topk, self.local_band_count, self.local_topk
        )
        key_freq_magnitude = self.select_key_frequencies(freq_magnitude, key_freq_indices)
        key_freq_phase = self.select_key_frequencies(freq_phase, key_freq_indices)
        key_freq = torch.polar(key_freq_magnitude, key_freq_phase)

        relation_matrix = self.Mahalanobis_mask(key_freq_magnitude, key_freq_phase)
        relation_matrix = relation_matrix.unsqueeze(0).unsqueeze(0)
        relation_freq = self.construct_key_freq_relationship(x, key_freq)

        relation_freq = relation_freq.reshape(B * C, -1, self.d_model)
        full_spectrum, _ = self.freq_attention(x=relation_freq.float(), attn_mask=relation_matrix)
        full_spectrum = full_spectrum.reshape(B, C, self.all_topk, self.d_model)

        key_freq_weights = F.softmax(key_freq_magnitude, dim=-1)
        key_freq_weights = key_freq_weights.unsqueeze(-1).expand_as(full_spectrum)
        full_spectrum = torch.sum(full_spectrum * key_freq_weights, dim=2)
        reconstructed_freq = torch.polar(full_spectrum, freq_phase)
        return reconstructed_freq

    def calculate_local_topk(self):
        return self.global_topk // self.local_band_count + 1

    def select_key_frequencies(self,freq_magnitude, key_freq_indices):
        B, C, freq_len = freq_magnitude.shape
        _, _, max_key_freqs = key_freq_indices.shape
        batch_indices = torch.arange(B).view(B, 1, 1).expand(B, C, max_key_freqs)
        channel_indices = torch.arange(C).view(1, C, 1).expand(B, C, max_key_freqs)
        key_freq_values = freq_magnitude[batch_indices, channel_indices, key_freq_indices]
        return key_freq_values

    def construct_key_freq_relationship(self,freq, key_freq):

        freq_magnitude = torch.abs(freq).mean(dim=-1, keepdim=True)
        normalized_magnitude = freq_magnitude / torch.sum(freq_magnitude, dim=-1, keepdim=True)
        freq = freq / normalized_magnitude
        key_freq_magnitude = torch.abs(key_freq).mean(dim=-1, keepdim=True)
        normalized_magnitude = key_freq_magnitude / torch.sum(key_freq_magnitude, dim=-1, keepdim=True)
        key_freq = key_freq / normalized_magnitude

        key_freq_expanded = key_freq.unsqueeze(-1)
        freq_magnitude_expanded = freq.unsqueeze(2)
        relationship_matrix = key_freq_expanded * freq_magnitude_expanded
        relationship_matrix =torch.sigmoid(relationship_matrix)

        return relationship_matrix

    def cumulative_energy_dynamic(self, freq_magnitude, global_topk, local_band_count, local_topk):
        B, C, freq_len = freq_magnitude.shape

        global_magnitude = (freq_magnitude**2).sum(dim=(0, 1))
        sorted_global_magnitude, sorted_global_indices = torch.sort(global_magnitude, descending=True)
        global_topk_indices = sorted_global_indices[:global_topk]

        if local_band_count <= 0:
            key_freq_indices = global_topk_indices
        else:
            band_size = freq_len // local_band_count
            bands = []
            for band in range(local_band_count):
                band_start = band * band_size
                band_end = (band + 1) * band_size if band != local_band_count - 1 else freq_len
                bands.append((band_start, band_end))

            local_indices = []
            for band_start, band_end in bands:
                band_magnitude = global_magnitude[band_start:band_end]
                band_sorted_indices = torch.argsort(band_magnitude, descending=True)
                local_topk_band_indices = band_sorted_indices[:local_topk] + band_start
                local_indices.append(local_topk_band_indices)

            local_topk_indices = torch.cat(local_indices)
            key_freq_indices = torch.cat([global_topk_indices, local_topk_indices],
                                         dim=0)
        key_freq_indices = key_freq_indices.unsqueeze(0).unsqueeze(0).repeat(B, C,
                                                                             1)
        return key_freq_indices


class Mahalanobis_mask(nn.Module):
    def __init__(self, topfreq):
        super(Mahalanobis_mask, self).__init__()
        self.A = nn.Parameter(torch.randn(topfreq, topfreq), requires_grad=True)
        self.w_amplitude = nn.Parameter(torch.tensor(0.5))
        self.w_phase = nn.Parameter(torch.tensor(0.5))
        self.projection_amplitude = nn.Linear(topfreq, topfreq)
        self.projection_phase = nn.Linear(topfreq, topfreq)

    def calculate_prob_distance(self, amplitude, phase):

        amplitude_diff = amplitude.unsqueeze(-1) - amplitude.unsqueeze(-2)
        phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        fused_diff = self.w_amplitude * amplitude_diff + self.w_phase * phase_diff

        temp = torch.einsum("dd,bxcd->bxcd", self.A, fused_diff)
        dist = torch.einsum("bxcd,bxcd->bcd", temp, temp)
        exp_dist = 1 / (dist + 1e-10)

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)

        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)
        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):

        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')

        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)

        return resample_matrix

    def forward(self, amplitude, phase):
        p = self.calculate_prob_distance(amplitude, phase)
        sample = self.bernoulli_gumbel_rsample(p)
        mask = sample.mean(dim=0)
        return mask


class FrequencyProcessingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, freq_len, kernel_size=2, down_sampling_window=None):
        super(FrequencyProcessingLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freq_len = freq_len
        self.kernel_size = kernel_size
        self.down_sampling_window = down_sampling_window
        self.weight = nn.Parameter(torch.ones(1,1,freq_len, dtype=torch.cfloat))
        self.low_weight = nn.Parameter(torch.tensor(0.9))

    def forward(self, freq):
        B, C, freq_len = freq.size()
        # Equivalent Sampling Rate
        esr = torch.sqrt(torch.tensor(min(self.kernel_size, self.out_channels / self.in_channels) /
                                      (self.down_sampling_window or 1.0)))

        nyquist_freq = esr / 2
        nyquist_idx = int(freq_len * nyquist_freq.item())
        mask = torch.zeros_like(freq, device=freq.device)

        mask[:, :, :nyquist_idx] = 1.0
        low_freq_mask  = mask
        high_freq_mask = 1 -  low_freq_mask
        low_freq = freq * low_freq_mask
        high_freq = freq * high_freq_mask

        weighted_low_freq = low_freq * self.weight
        weighted_high_freq = high_freq * self.weight

        low_weight = torch.sigmoid(self.low_weight)
        high_weight = 1 - low_weight

        mixed_freq = low_weight * weighted_low_freq + high_weight * weighted_high_freq
        return mixed_freq

