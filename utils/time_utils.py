import torch
import torch.nn as nn
import tinycudann as tcnn


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": i,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        N_freqs = self.kwargs["num_freqs"]

        encoding_config = {"otype": "Frequency", "n_frequencies": N_freqs}
        tcnn_encoding = tcnn.Encoding(
            n_input_dims=d,
            encoding_config=encoding_config,
        )
        embed_fns.append(tcnn_encoding)
        out_dim += tcnn_encoding.n_output_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=4, is_2dgs=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6
        self.t_multires = 1
        self.skips = []

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        time_encoding_config = {"otype": "Frequency", "n_frequencies": self.t_multires}
        pos_encoding_config = {"otype": "Frequency", "n_frequencies": multires}
        self.embed_time_fn = tcnn.Encoding(
            n_input_dims=1, encoding_config=time_encoding_config
        )
        self.embed_fn = tcnn.Encoding(
            n_input_dims=3, encoding_config=pos_encoding_config
        )
        self.input_ch = self.embed_fn.n_output_dims + self.embed_time_fn.n_output_dims

        self.is_2dgs = is_2dgs
        self.out_size = 10
        self.pos_offset = 3
        self.rot_offset = self.pos_offset + 4
        self.scale_offset = self.rot_offset + 2 if is_2dgs else self.rot_offset + 3
        self.bound_size = 1.0

        network_config = {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": W,
            "n_hidden_layers": D,
        }

        self.linear = tcnn.Network(
            n_input_dims=self.input_ch,
            n_output_dims=self.out_size,
            network_config=network_config,
        )

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x / self.bound_size)
        h = torch.cat([x_emb, t_emb], dim=-1)
        output = self.linear(h)

        d_xyz = output[:, 0 : self.pos_offset]
        d_xyz = d_xyz * self.bound_size

        return (
            d_xyz,
            output[:, self.pos_offset : self.rot_offset],
            output[:, self.rot_offset : self.scale_offset],
        )
