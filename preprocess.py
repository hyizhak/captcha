import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths for input and output directories
input_dir = "train"
output_dir = "processed"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def remove_lines(image):
    # Convert the image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    img_array = np.array(gray_image)

    height, width = img_array.shape
    result = img_array.copy()  # Create a copy to modify

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            count = 0
            pixel_value = img_array[y, x]

            # Check surrounding pixels
            if pixel_value == img_array[y - 1, x + 1]:
                count += 1
            if pixel_value == img_array[y, x + 1]:
                count += 1
            if pixel_value == img_array[y + 1, x + 1]:
                count += 1
            if pixel_value == img_array[y - 1, x]:
                count += 1
            if pixel_value == img_array[y + 1, x]:
                count += 1
            if pixel_value == img_array[y - 1, x - 1]:
                count += 1
            if pixel_value == img_array[y, x - 1]:
                count += 1
            if pixel_value == img_array[y + 1, x - 1]:
                count += 1

            # If the count is low, consider it noise and set it to a new value (e.g., white)
            if count <= 3 and count > 0:
                result[y, x] = 255  # Set to white to remove interference

    return result

def tokenize_contours(image):
    # Find contours of each character
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small contours by size
        if w * h > 100 and w > 3 and h > 3:
            filtered_contours.append((contour, x, y, w, h))
    
    # Sort contours from left to right based on x-coordinate
    filtered_contours = sorted(filtered_contours, key=lambda c: c[1])

    return filtered_contours

def tokenize_watershed(image, binary):
    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,  kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image, markers)

    # return the box based on the markers
    boxes = []
    for i in range(1, markers.max() + 1):
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == i] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        boxes.append((0, x, y, w, h))
    return boxes

def tokenize_projection(image, step=3):
    # Sum of white pixels along each column (for vertical segmentation)
    column_sums = np.sum(image, axis=0)

    # Find peaks in the column sums to detect character boundaries
    threshold = 0.2 * np.max(column_sums)
    start = None
    segments = []

    # Loop through the column sums with a step size
    for i in range(0, len(column_sums), step):
        sum_val = column_sums[i]
        
        if sum_val > threshold and start is None:
            start = i  # Start of a new character segment
        elif sum_val <= threshold and start is not None:
            segments.append((0, start, 0, i - start, image.shape[1]))  # End of a character segment
            start = None

    return segments

def process_image(file_path, file_name, charcount, tokenizor, output_dir):
    colored_img = cv2.imread(file_path)
    # Load the captcha image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    # Remove noisy lines
    img = remove_lines(img)

    # plt.imshow(img, cmap="gray")
    # plt.show()
    
    # Threshold the image to binary (black and white)
    _, processed_img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    # _, processed_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plt.imshow(processed_img, cmap="gray")
    # plt.show()

    # Invert the image to make the background white and the letters black
    processed_img = cv2.bitwise_not(processed_img)

    # Apply dilation to connect nearby parts of letters
    kernel = np.ones((2, 2), np.uint8)  # Adjust kernel size based on your captcha structure
    dilated_img = cv2.dilate(processed_img, kernel, iterations=1)

    # plt.imshow(dilated_img, cmap="gray")
    # plt.show()

    if tokenizor == 'contours':
        filtered_contours = tokenize_contours(dilated_img) 
    elif tokenizor == 'projection':
        filtered_contours = tokenize_projection(dilated_img)

    # Extract characters and save with the naming convention
    captcha_text = file_name.split("-")[0]  # Assuming filename format is 'text-0.png'

    charcount['total'] = charcount.get('total', 0) + len(captcha_text)

    dilation_steps = range(5)

    if len(captcha_text) != len(filtered_contours):
        # Apply dilation to connect nearby parts of letters
        for i in dilation_steps:
            dilated_img = cv2.dilate(processed_img, kernel, iterations=i)
            if tokenizor == 'contours':
                filtered_contours = tokenize_contours(dilated_img)
                if len(captcha_text) == len(filtered_contours):
                    break
            elif tokenizor == 'projection':
                for step in range(1, 5):
                    filtered_contours = tokenize_projection(dilated_img, step)
                    if len(captcha_text) == len(filtered_contours):
                        break

    if len(captcha_text) != len(filtered_contours):
        plt.imshow(dilated_img, cmap="gray")
        plt.show()
        print(f"Skipping {file_name} due to length mismatch: {len(captcha_text)} != {len(filtered_contours)}")
        return
    
    for i, (contour, x, y, w, h) in enumerate(filtered_contours):
        # Crop and resize each character
        char_img = processed_img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (224, 224))  # Resize to a fixed size
        
        # Save each character with the naming convention, mark the number of the character
        charcount[captcha_text[i]] = charcount.get(captcha_text[i], 0) + 1
        char_filename = f"{captcha_text[i]}_{charcount[captcha_text[i]]}.png"
        cv2.imwrite(os.path.join(output_dir, char_filename), char_img)
    

if __name__ == "__main__":
    # Process each file in the input directory
    tokenizer = 'projection'
    output_dir = f"{output_dir}_{tokenizer}"
    charcount = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            process_image(os.path.join(input_dir, filename), filename, charcount)

    print(charcount['total'] / len(os.listdir(output_dir)))