import torch
# from grace_dl.torch import Compressor

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return indices, values

class RandomKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx):
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)