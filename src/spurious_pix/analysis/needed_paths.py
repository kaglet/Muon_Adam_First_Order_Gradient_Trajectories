from pathlib import Path

__all__ = ["sp_pix_outputs_dir"]

sp_pix_outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs/spurious_pixel"

if __name__ == "__main__":
    print("Spurious pixel outputs dir verified as: ", sp_pix_outputs_dir) 