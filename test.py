import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import os

# カスケード分類器が既にあれば読み込み、無ければ作成する
if (os.path.exists('face_recognition_cascade.csv')):
    loaded_cascade = np.loadtxt('face_recognition_cascade.csv')
    # 撮影した写真の人物と一致するか照合
else:
    

# read saved human face picture
known_face_imgs_path = ["known-face_01.jpg", "known-face_02.jpg", "known-face_03.jpg"]
known_face_imgs = []
for path in known_face_imgs_path:
    img = fr.load_image_file(path)
    known_face_imgs.append(img)
    
face_img_to_check = fr.load_image_file("face_to_check.jpg")

# inspect human face area
known_face_locs = []
for img in known_face_imgs:
    loc = fr.face_locations(img, model="hog")
    known_face_locs.append(loc)
    
face_loc_to_check = fr.face_locations(face_img_to_check, model="hog")

def draw_face_locations(img, locations):
    """
    example
    locations: [(139, 262, 325, 77)]
    """
    fig, axes = plt.subplots()
    axes.imshow(img)
    axes.set_axis_off()
    for i, (top, right, bottom, left) in enumerate(locations):
        w, h = right - left, bottom - top
        axes.add_patch(plt.Rectangle((left, top), w, h, ec="r", lw=2, fill=None))
    plt.show()
    
# for img, loc in zip(known_face_imgs, known_face_locs):
#     draw_face_locations(img, loc)
    
# draw_face_locations(face_img_to_check, face_loc_to_check)

known_face_encodings = []  # known_face_encodings: attribute
for img, loc in zip(known_face_imgs, known_face_locs):
    (encoding,) = fr.face_encodings(img, loc)
    known_face_encodings.append(encoding)

# extracting array by split assignment
(face_encoding_to_check, ) = fr.face_encodings(face_img_to_check, face_loc_to_check)  # (face_encoging_to_check,): attribute

# print(type(face_encoding_to_check))  # [array([], )]
# print(face_encoding_to_check.shape)  # (128,)

# np.savetxt('face_recognition_cascade.csv', known_face_encodings)

matches = fr.compare_faces(known_face_encodings, face_encoding_to_check)
print(matches)