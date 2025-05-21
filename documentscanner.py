import cv2
from util import preprocess_image, get_contours, warp_perspective

def scan_document(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    edged = preprocess_image(image)

    contour = get_contours(edged)
    if contour is None:
        print("No document found.")
        return

    scanned = warp_perspective(orig, contour)
    
    # Optional post-processing: convert to grayscale and threshold
    gray_scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    scanned_final = cv2.adaptiveThreshold(
        gray_scanned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    cv2.imshow("Scanned", scanned_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    scan_document("test_image.jpg")
