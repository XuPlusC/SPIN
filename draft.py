import cv2

if __name__ == "__main__":
    img = cv2.imread("/home/pose/Pictures/origin.jpg")
    cv2.imshow("wee", img)
    cv2.waitKey(0)