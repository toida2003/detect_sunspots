import glob
import cv2
import copy


def main():
    files = glob.glob("sun_imgs/*.jpg", recursive=True)

    for i, img_path in enumerate(files):
        img_raw = cv2.imread(img_path)
        clip_range = 100
        img_raw = img_raw[
            clip_range : img_raw.shape[0] - clip_range,
            clip_range : img_raw.shape[1] - clip_range,
        ]
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        img_bin = cv2.adaptiveThreshold(
            img_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            20,
        )
        img_bin = cv2.medianBlur(img_bin, 5)

        contours, _ = cv2.findContours(
            ~img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        x_list = []
        y_list = []
        for pos in contours[0]:
            x_list.append(pos[0][0])
            y_list.append(pos[0][1])
        upper = min(y_list)
        lower = max(y_list)
        left = min(x_list)
        right = max(x_list)

        img_clip = img_bin[upper:lower, left:right]
        img_clip_color = img_raw[upper:lower, left:right]

        contours, hierarchy = cv2.findContours(
            img_clip, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        sunspots = []
        for contour, h in zip(contours, hierarchy[0]):
            if h[3] != -1:
                sunspots.append(contour)

        img_sunspots = copy.deepcopy(img_clip_color)
        for sunspot in sunspots:
            if sunspot.shape[0] > 4:
                img_sunspots = cv2.drawContours(
                    img_sunspots, sunspot, -1, (255, 0, 0), 2
                )

        cv2.imwrite(f"sun_imgs_rename/{i}.jpg", img_clip_color)
        cv2.imwrite(f"result/result_bin_{i}.jpg", img_bin)
        cv2.imwrite(f"result/result_{i}.jpg", img_sunspots)


if __name__ == "__main__":
    main()
