import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import FastICA

def normalize_for_display(channel):
    # Scales a mathematical array to 0-255 for standard image viewing
    channel_min = np.min(channel)
    channel_max = np.max(channel)
    normalized = (channel - channel_min) / (channel_max - channel_min)
    return (normalized * 255).astype(np.uint8)

def demonstrate_rgb_ica(image_paths, viz_paths):
    # 1. Load two sources as grayscale and flatten
    sources = []
    viz_images = []
    for path in image_paths[:2]:
        img = Image.open(path).convert("RGB")
        # PIL resize is (width, height)
        img = img.resize((500, 300)) 
        # NumPy shape becomes (height, width, channels) -> (500, 300, 3)
        # Transpose moves channels to the front -> (3, 500, 300)
        img_array = np.array(img).astype(np.float64).transpose(2, 0, 1)
        sources.append(img_array.reshape(3, -1))

    for path in viz_paths:
        img = Image.open(path).convert("RGB")
        # PIL resize is (width, height)
        img = img.resize((400, 300)) 
        # NumPy shape becomes (height, width, channels) -> (500, 300, 3)
        # Transpose moves channels to the front -> (3, 500, 300)
        img_array = np.array(img).astype(np.float64)
        viz_images.append(img_array)

    # Shape: (6, 150000)
    noise_img = np.random.normal(128, 128, size=(300, 500, 3)).clip(0, 255)
    noise_img_l = noise_img.astype(np.float64).transpose(2, 0, 1)
    noise_img_l = noise_img_l.reshape(3, -1)
    sources.append(noise_img_l)
    S = np.vstack(sources)

    # Define Mixing Matrices
    A = np.array([
        [0.33, 0.0, 0.0, 0.33, 0.0, 0.0, 0.33, 0, 0],
        [0.0, 0.33, 0.0, 0.0, 0.33, 0.0, 0, 0.33, 0],
        [0.0, 0.0, 0.33, 0.0, 0.0, 0.33, 0, 0, 0.33]
    ])
    B_raw = np.random.rand(3, 9)
    row_sums = B_raw.sum(axis=1, keepdims=True)
    A_normalized = B_raw / row_sums
    B = np.round(A_normalized, 2)
                
    # 3. Generate the 3 mixed channels
    # Shape: (3, 150000)
    X = np.dot(B, S)

    # 4. Correctly rebuild the spatial image
    # Reshape to (Channels, Height, Width) THEN transpose to (Height, Width, Channels)
    mixed = X.reshape(3, 300, 500).transpose(1, 2, 0)
    source1_reconstructed = S[0:3].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)
    source2_reconstructed = S[3:6].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)


    A_pseudo_inverse = np.linalg.pinv(A)
    B_pseudo_inverse = np.linalg.pinv(B)

    S_recovered_clean = np.dot(B_pseudo_inverse, X)
    rec_source1_clean = S_recovered_clean[0:3].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)
    rec_source2_clean = S_recovered_clean[3:6].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)
    rec_source3_clean = S_recovered_clean[6:9].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)

    S_recovered_messy = np.dot(A_pseudo_inverse, X)
    rec_source1_messy = S_recovered_messy[0:3].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)
    rec_source2_messy = S_recovered_messy[3:6].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)
    rec_source3_messy = S_recovered_messy[6:9].reshape(3, 300, 500).transpose(1, 2, 0).astype(np.uint8)


    # 6. Plotting
    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 10))

    # Top Row: The Mixing Process
    axes1[0, 0].imshow(source1_reconstructed)
    axes1[0, 0].set_title("Source A", fontsize=18, fontweight="bold")
    axes1[0, 0].axis("off")

    axes1[0, 1].imshow(source2_reconstructed)
    axes1[0, 1].set_title("Source B", fontsize=18, fontweight="bold")
    axes1[0, 1].axis("off")

    axes1[0, 2].imshow(noise_img.astype(np.uint8))
    axes1[0, 2].set_title("Noise", fontsize=18, fontweight="bold")
    axes1[0, 2].axis("off")

    axes1[1, 1].imshow(mixed.astype(np.uint8))
    axes1[1, 1].set_title("Observed Image", fontsize=18, fontweight="bold")
    axes1[1, 1].axis("off")

    axes1[1, 0].axis("off")
    axes1[1, 2].axis("off")

    fig1.tight_layout()

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 10))

    # Top Row: The Mixing Process
    axes2[0, 0].axis("off")

    axes2[0, 1].imshow(mixed.astype(np.uint8))
    axes2[0, 1].set_title("Observed Image", fontsize=18, fontweight="bold")
    axes2[0, 1].axis("off")

    axes2[0, 2].axis("off")

    axes2[1, 0].imshow(rec_source1_clean)
    axes2[1, 0].set_title("Noise", fontsize=18, fontweight="bold")
    axes2[1, 0].axis("off")

    axes2[1, 1].imshow(rec_source2_clean)
    axes2[1, 1].set_title("Observed Image", fontsize=18, fontweight="bold")
    axes2[1, 1].axis("off")

    axes2[1, 2].imshow(rec_source3_clean)
    axes2[1, 2].set_title("Observed Image", fontsize=18, fontweight="bold")
    axes2[1, 2].axis("off")

    fig2.tight_layout()


    fig1.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/sourcemix.png", dpi=300, bbox_inches="tight")
    fig2.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/sourcesep.png", dpi=300, bbox_inches="tight")


    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 10))

    # Top Row: The Mixing Process
    axes3[0, 0].axis("off")

    axes3[0, 1].imshow(viz_images[0].astype(np.uint8))
    axes3[0, 1].set_title("Observed Image", fontsize=18, fontweight="bold")
    axes3[0, 1].axis("off")

    axes3[0, 2].axis("off")

    axes3[1, 0].imshow(viz_images[1].astype(np.uint8))
    axes3[1, 0].set_title("Background", fontsize=18, fontweight="bold")
    axes3[1, 0].axis("off")

    axes3[1, 1].imshow(viz_images[2].astype(np.uint8))
    axes3[1, 1].set_title("Neurons", fontsize=18, fontweight="bold")
    axes3[1, 1].axis("off")

    axes3[1, 2].imshow(viz_images[3].astype(np.uint8))
    axes3[1, 2].set_title("Noise", fontsize=18, fontweight="bold")
    axes3[1, 2].axis("off")

    fig3.tight_layout()
    fig3.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/real_sourcesep.png", dpi=300, bbox_inches="tight")

# Example usage:
# demonstrate_rgb_ica(["bone_fracture.jpg", "sheep.jpg"])

if __name__ == "__main__":
    img = [
        "/home/jaschneider/projects/DRG-CaSeg/2024_04-19_FSNY_Margaretta_lamb_LH_4565-scaled.png",
        "/home/jaschneider/projects/DRG-CaSeg/JPG-2024_aof_ao_center_winter_davos_9947.jpg"
    ]
    viz = [
        "/home/jaschneider/projects/DRG-CaSeg/1st A2_t00001.jpg",
        "/home/jaschneider/projects/DRG-CaSeg/background_component.png",
        "/home/jaschneider/projects/DRG-CaSeg/neuron_component.png",
        "/home/jaschneider/projects/DRG-CaSeg/noise_component.png"
    ]
    demonstrate_rgb_ica(img, viz)