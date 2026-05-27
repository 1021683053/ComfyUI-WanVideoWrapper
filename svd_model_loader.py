import torch
from safetensors import safe_open


class SVDModelLoader:
    @staticmethod
    def load_svd_compressed_model(model_path, device="cpu"):
        sd = {}
        compressed_keys = set()

        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            for key in keys:
                if key.endswith(".A"):
                    compressed_keys.add(key[:-2])

            print(f"Loading SVD compressed model: {len(compressed_keys)} compressed weights")

            for base_key in compressed_keys:
                a_key = f"{base_key}.A"
                b_key = f"{base_key}.B"
                shape_key = f"{base_key}.shape"

                if a_key in keys and b_key in keys:
                    a = f.get_tensor(a_key)
                    b = f.get_tensor(b_key)
                    original_dtype = a.dtype

                    a_float = a.float().to(device)
                    b_float = b.float().to(device)

                    reconstructed = torch.mm(a_float, b_float).cpu().to(original_dtype)

                    if shape_key in keys:
                        original_shape = tuple(f.get_tensor(shape_key).tolist())
                        if len(original_shape) > 2:
                            reconstructed = reconstructed.reshape(original_shape)

                    sd[base_key] = reconstructed

                    if device == "cuda":
                        torch.cuda.empty_cache()

            for key in keys:
                if not (key.endswith(".A") or key.endswith(".B") or key.endswith(".shape")):
                    if key not in sd:
                        sd[key] = f.get_tensor(key)

        print(f"SVD model loaded: {len(sd)} total tensors")
        return sd

    @staticmethod
    def is_svd_compressed(model_path):
        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                return any(key.endswith(".A") for key in f.keys())
        except Exception:
            return False
