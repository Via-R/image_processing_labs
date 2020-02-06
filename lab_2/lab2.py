import cv2
import numpy as np

a = 50
b = 50

def apply_operation_to_img(op, img):
    return list(map(lambda x: list(map(op, x)), img))

def zero_log10(x):
    return np.log10(x) if x > 0 else 0

def main():
    global a, b

    img = cv2.imread('moon.jpg', 0)
    print("Image read")
    f_img = np.fft.fft2(img).tolist()
    print("Image transformed")
    f_abs_img = apply_operation_to_img(abs, f_img)
    f_angle = apply_operation_to_img(np.angle, f_img)
    print("Image broke to complex pieces")
    log_abs_img = apply_operation_to_img(lambda z: a + b * zero_log10(z), f_abs_img)
    print("Moduli processed by logarithm")
    result = np.multiply(log_abs_img, np.exp(np.multiply(f_angle, 1j)))
    # print(result)
    print("Complex parts combined")
    output = np.real(np.fft.ifft2(result))
    print("Transformation reversed")
    print(output)
    cv2.imshow("Input", img)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
