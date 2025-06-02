import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image
import math
import os
import cv2
import glob # For more flexible file loading

# --- Configuration ---
# All configurable parameters are defined here for easy modification.
CONFIG = {
    "image_compression": {
        "lena_image_path": "img/lena.jpg",
        "output_compression_image_path": "img/svd_compression.png",
        "output_norm_plot_path": "img/norm_vs_k.png",
        "k_values": [1, 10, 50, 100, 150, 200, 250, 300, 350],
        "figure_size_compression": (15, 5),
        "figure_size_norm_plot": (10, 5)
    },
    "facial_recognition": {
        "train_folder": "trainface",
        "test_folder": "testface",
        "output_training_images_path": "training_images.png",
        "output_test_results_prefix": "test_image_result_",
        "image_resize_dims": (56, 46), # (width, height)
        "recognition_epsilon": 1000, # Threshold for recognition (adjust as needed)
        "figure_size_display": (20, 10),
        "figure_size_recognition_result": (10, 5)
    }
}

# --- SVD Compression Functions ---
def svd_compress(U, S, V, k):
    """
    Performs Singular Value Decomposition (SVD) compression on an image channel.

    Args:
        U (numpy.ndarray): Left singular vectors.
        S (numpy.ndarray): Singular values.
        V (numpy.ndarray): Right singular vectors (transposed).
        k (int): Number of singular values to keep for compression.

    Returns:
        numpy.ndarray: The compressed image channel.
    """
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    p_svd = np.dot(U_k, np.dot(S_k, V_k))
    return p_svd

# --- Facial Recognition Functions ---
def load_images_from_folder(folder_path, resize_dims, file_pattern="*.jpg"):
    """
    Loads grayscale images from a specified folder, resizes them, and flattens them.
    Includes robust error handling for folder existence and image loading.

    Args:
        folder_path (str): The path to the folder containing images.
        resize_dims (tuple): A tuple (width, height) for resizing images.
        file_pattern (str): The pattern to match image files (e.g., "*.jpg", "*.png").

    Returns:
        list: A list of flattened (1D) image arrays.
    """
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please create it and add images.")
        return []

    # Use glob for more robust file pattern matching
    image_files = glob.glob(os.path.join(folder_path, file_pattern))
    if not image_files:
        print(f"Warning: No image files found in '{folder_path}' matching pattern '{file_pattern}'.")
        return []

    for img_file in image_files:
        try:
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image '{img_file}'. Skipping.")
                continue
            
            img_resized = cv2.resize(img, resize_dims) # Resize to (width, height)
            images.append(img_resized.flatten())
        except Exception as e:
            print(f"Error processing image '{img_file}': {e}. Skipping.")
            continue
    return images

def compute_svd_for_faces(images):
    """
    Computes the SVD for a set of images to find eigenfaces.

    Args:
        images (numpy.ndarray): A 2D array where each row is a flattened image.

    Returns:
        tuple: U, S, Vt (from SVD), and the mean face.
    """
    if images.size == 0:
        print("Error: Cannot compute SVD with an empty image set.")
        return None, None, None, None
    
    mean_face = np.mean(images, axis=0)
    A = images - mean_face
    U, S, Vt = la.svd(A, full_matrices=False)
    return U, S, Vt, mean_face

def project_faces(images, mean_face, Vt):
    """
    Projects a set of images into the eigenface space.

    Args:
        images (numpy.ndarray): A 2D array where each row is a flattened image.
        mean_face (numpy.ndarray): The mean face of the training set.
        Vt (numpy.ndarray): The transposed singular vectors (eigenfaces).

    Returns:
        numpy.ndarray: The projected faces in the eigenface space.
    """
    if images.size == 0 or mean_face is None or Vt is None:
        print("Error: Invalid input for projecting faces.")
        return np.array([])
        
    A = images - mean_face
    return np.dot(A, Vt.T)

def recognize_face(test_face, projected_faces, epsilon0):
    """
    Recognizes a test face by finding the closest match in the projected faces.

    Args:
        test_face (numpy.ndarray): The projected test face.
        projected_faces (numpy.ndarray): A 2D array of projected training faces.
        epsilon0 (float): The threshold for recognition.

    Returns:
        tuple: (closest_face_id, recognized_status)
               closest_face_id (int): The index of the closest training face.
               recognized_status (bool): True if recognized, False otherwise.
    """
    if test_face is None or projected_faces.size == 0:
        print("Error: Invalid input for face recognition.")
        return -1, False

    # Calculate Euclidean distance between the test face and all projected faces
    norm_distances = np.linalg.norm(projected_faces - test_face, axis=1)
    closest_face_id = np.argmin(norm_distances)
    min_distance = norm_distances[closest_face_id]
    print(f'Min Distance: {min_distance:.2f}') # Format for cleaner output
    return closest_face_id, min_distance <= epsilon0

# --- Main Script Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('img', exist_ok=True)

    # --- Part 1: SVD Image Compression Demonstration ---
    print("--- SVD Image Compression Demonstration ---")
    
    lena_path = CONFIG["image_compression"]["lena_image_path"]
    if not os.path.exists(lena_path):
        print(f"Error: Lena image not found at '{lena_path}'. Skipping image compression demonstration.")
    else:
        try:
            img = Image.open(lena_path)
            img_array = np.array(img)

            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            U_r, S_r, V_r = la.svd(r, full_matrices=False)
            U_g, S_g, V_g = la.svd(g, full_matrices=False)
            U_b, S_b, V_b = la.svd(b, full_matrices=False)

            norm_list = {}
            plt.figure(figsize=CONFIG["image_compression"]["figure_size_compression"])
            k_list = CONFIG["image_compression"]["k_values"]

            plt.subplot(2, 5, 1)
            plt.imshow(img_array)
            plt.title('Original Image')
            plt.axis('off')
            i = 1

            for k in k_list:
                r_svd = svd_compress(U_r, S_r, V_r, k)
                g_svd = svd_compress(U_g, S_g, V_g, k)
                b_svd = svd_compress(U_b, S_b, V_b, k)

                # Stack channels to form the compressed color image
                img_svd = np.zeros((r_svd.shape[0], r_svd.shape[1], 3))
                img_svd[:, :, 0] = r_svd
                img_svd[:, :, 1] = g_svd
                img_svd[:, :, 2] = b_svd

                img_svd = np.clip(img_svd, 0, 255).astype(np.uint8)
                norm_list[k] = np.linalg.norm(img_array - img_svd) # Calculate Frobenius norm

                plt.subplot(2, 5, i + 1)
                plt.imshow(img_svd)
                plt.title(f'k = {k}')
                plt.axis('off')
                i += 1

            plt.tight_layout()
            plt.savefig(CONFIG["image_compression"]["output_compression_image_path"])
            plt.show()

            # Plot the norm vs k
            plt.figure(figsize=CONFIG["image_compression"]["figure_size_norm_plot"])
            plt.plot(list(norm_list.keys()), list(norm_list.values()))
            plt.xlabel('k')
            plt.ylabel('Norm (Compression Error)')
            plt.title('Norm vs k (Compression Error)')
            plt.grid(True)
            plt.savefig(CONFIG["image_compression"]["output_norm_plot_path"])
            plt.show()
        except Exception as e:
            print(f"An error occurred during SVD Image Compression: {e}")


    print("\n--- Facial Recognition using SVD ---")
    # --- Part 2: Facial Recognition using SVD ---
    train_folder = CONFIG["facial_recognition"]["train_folder"]
    test_folder = CONFIG["facial_recognition"]["test_folder"]
    resize_dims = CONFIG["facial_recognition"]["image_resize_dims"]
    epsilon0 = CONFIG["facial_recognition"]["recognition_epsilon"]

    # Load training and testing images with robust loading
    train_images_list = load_images_from_folder(train_folder, resize_dims)
    test_images_list = load_images_from_folder(test_folder, resize_dims)

    if not train_images_list:
        print("Facial recognition skipped: No training images loaded.")
    elif not test_images_list:
        print("Facial recognition skipped: No test images loaded.")
    else:
        train_images = np.array(train_images_list)
        test_images = np.array(test_images_list)

        # Display all training images
        plt.figure(figsize=CONFIG["facial_recognition"]["figure_size_display"])
        num_train_images = len(train_images)
        # Adjust subplot grid based on the number of training images
        rows = math.ceil(num_train_images / 4) # Assuming 4 images per row
        for i, img_data in enumerate(train_images):
            plt.subplot(rows, 4, i+1) 
            plt.imshow(img_data.reshape(resize_dims[1], resize_dims[0]), cmap='gray')
            plt.title(f'Train Image {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(CONFIG["facial_recognition"]["output_training_images_path"])
        plt.show()

        # Compute SVD for facial recognition
        U, S, Vt, mean_face = compute_svd_for_faces(train_images)

        if mean_face is None: # Check if SVD computation was successful
            print("Facial recognition skipped: SVD computation failed.")
        else:
            # Project training faces into the eigenface space
            projected_train_faces = project_faces(train_images, mean_face, Vt)
            
            # Recognize each test face
            test_img_count = 0
            for test_img_data in test_images:
                # Project the single test image
                projected_test_face = project_faces(np.array([test_img_data]), mean_face, Vt)[0]

                # Recognize the face
                id, recognized = recognize_face(projected_test_face, projected_train_faces, epsilon0)
                print(f'Test Image {test_img_count+1}: Recognized = {recognized}, Closest Match ID = {id}')

                # Display the test image and the closest match
                plt.figure(figsize=CONFIG["facial_recognition"]["figure_size_recognition_result"])
                plt.subplot(1, 2, 1)
                plt.imshow(test_img_data.reshape(resize_dims[1], resize_dims[0]), cmap='gray')
                plt.title('Test Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(train_images[id].reshape(resize_dims[1], resize_dims[0]), cmap='gray')
                plt.title(f'Match: {recognized}\nID: {id}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{CONFIG["facial_recognition"]["output_test_results_prefix"]}{test_img_count}.png')
                plt.show()

                test_img_count += 1