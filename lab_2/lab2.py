import cv2
import numpy as np

# Moduli for log transform
a = 50
b = 50

# Power constant for pow transform
alpha = 0.5

# Dictionary of available transformations
transformations = {
    "log": lambda z: a + b * zero_log10(z),
    "pow": lambda z: pow(z, alpha)
}

def apply_operation_to_img(op, img):
    '''Applies a given operation to all elements of a two-dimensional list.'''

    # Processing rgb image
    # return list(map(lambda x: list(map(lambda y: list(map(op, y)), x)), img))

    # Processing grayscale image
    return list(map(lambda x: list(map(op, x)), img))

def zero_log10(x):
    '''Returns log10 of x if it's greater than 0, otherwise returns 0'''

    return np.log10(x) if x > 0 else 0

def main(transform_type="pow"):
    '''Main function, receives transformation type as an argument.'''

    # Read the image
    img = cv2.imread('moon.jpg', 0)
    print("Image read")

    # Calculates Fourier transform of the given image
    f_img = np.fft.fft2(img).tolist()
    print("Image transformed")

    # Calculates moduli and angles of image elements, saves them as two 2d lists
    f_abs_img = apply_operation_to_img(abs, f_img)
    f_angle = apply_operation_to_img(np.angle, f_img)
    print("Image broke to complex pieces")

    # Choose one specific transformation
    mat_transform = transformations[transform_type]

    # Perform the chosen transform over moduli of complex matrix
    transformed_abs_img = apply_operation_to_img(mat_transform, f_abs_img)
    print("Moduli processed by logarithm")
    
    # Combine moduli with angles to get complex matrix back
    result = np.multiply(transformed_abs_img, np.exp(np.multiply(f_angle, 1j)))
    print("Complex parts combined")

    # Calculate inverted Fourier transform and leave real part only
    output = np.real(np.fft.ifft2(result))
    print("Transformation reversed")

    # Show the results
    cv2.imshow("Input", img)
    cv2.imshow(f"Output {transform_type}", output)

if __name__ == "__main__":
    main("log")
    main("pow")
    cv2.waitKey(0)
