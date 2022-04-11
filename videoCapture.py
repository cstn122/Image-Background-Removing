# Ref: https://www.codegrepper.com/code-examples/python/opencv+save+image+python+from+videocapture
# 連拍，適用於收集相機校正用dataset時

import cv2
vidcap = cv2.VideoCapture(1) # USB camera
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("yframe%d.jpg" %count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    print('Read a new frame: ', success)
    img = cv2.imread("/captured/yframe%d.jpg" % count)
    var_x = 0
    dir = 'l'
    count += 1

    for j in range(2):
        # 裁切區域的 x 與 y 座標（左上角）
        x = 0 + var_x
        y = 0
        # 裁切區域的長度與寬度
        w = 672
        h = 397
        # 裁切圖片
        crop_img = img[y:y + h, x:x + w]
        # 寫入圖檔
        cv2.imwrite('cropped_%s/ycropped_0_%s.jpg' %(dir, dir), crop_img)
        var_x = 672
        dir = 'r'