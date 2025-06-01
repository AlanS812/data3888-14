import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from PIL import Image, ImageFilter  # ← add ImageFilter

# Load & resize
img_path = 'cell_25660_100.png'
image = Image.open(img_path)
image_resized = image.resize((224, 224))
image_array = np.asarray(image_resized)

# --- apply Gaussian blur then noise to this one image ---
pil_img        = Image.fromarray(image_array)
blurred        = pil_img.filter(ImageFilter.GaussianBlur(radius=19))
image_blurred  = np.array(blurred)
image = np.array(pil_img)

np.random.seed(3888)
noise          = np.random.normal(0, 30, image_blurred.shape)
image_noisy    = np.clip(image_blurred + noise, 0, 255).astype(np.uint8)


# Flatten the image
image_flat = image.flatten().reshape(1,-1)
image_flat_aug = image_noisy.flatten().reshape(1, -1)

# Load the fitted PCA model
pca = joblib.load('../app/Base_pca.joblib')  # Replace with your actual path

# Transform and inverse transform the image using PCA
image_pca = pca.transform(image_flat)
image_reconstructed = pca.inverse_transform(image_pca)
image_reconstructed = image_reconstructed.reshape(image_array.shape)

# Normalize for display
def normalize(img):
    img = img - img.min()
    return img / img.max()

# Compute PCA reconstruction for the augmented (noisy) image
image_pca_aug = pca.transform(image_flat_aug)
image_reconstructed_aug = pca.inverse_transform(image_pca_aug).reshape(image_array.shape)

# Plot original, its reconstruction, augmented, and its reconstruction side by side
plt.figure(figsize=(8, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(normalize(image), cmap='gray')
plt.title("Original")
plt.axis('off')

# PCA Reconstructed (original)
plt.subplot(2, 2, 2)
plt.imshow(normalize(image_reconstructed), cmap='gray')
plt.title("PCA Reconstructed")
plt.axis('off')

# Augmented (noisy)
plt.subplot(2, 2, 3)
plt.imshow(normalize(image_noisy), cmap='gray')
plt.title("Augmented")
plt.axis('off')

# PCA Reconstructed (augmented)
plt.subplot(2, 2, 4)
plt.imshow(normalize(image_reconstructed_aug), cmap='gray')
plt.title("PCA Reconstructed\n(Augmented)")
plt.axis('off')

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=400)
# plt.show()

plt.figure(figsize=(10, 4))
for i in range(5):
    ax = plt.subplot(2, 5, i + 1)
    pc_img = pca.components_[i].reshape(image_array.shape)
    plt.imshow(normalize(pc_img), cmap='gray')
    ax.set_title(f'PC {i+1}')
    ax.axis('off')
plt.suptitle("Top 10 Principal Components", y=1.02)
plt.tight_layout()
plt.savefig('components.png, dpi=400')
plt.show()

