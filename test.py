face_img_to_check = fr.load_image_file("face_to_check.jpg")
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
    
# for img, loc in zip(known_face_imgs, registered_face_locs):
#     draw_face_locations(img, loc)
    
# draw_face_locations(face_img_to_check, face_loc_to_check)

known_face_encodings = []  # known_face_encodings: attribute

# extracting array by split assignment
(face_encoding_to_check, ) = fr.face_encodings(face_img_to_check, face_loc_to_check)  # (face_encoging_to_check,): attribute

# print(type(face_encoding_to_check))  # [array([], )]
# print(face_encoding_to_check.shape)  # (128,)

# np.savetxt('face_recognition_cascade.csv', known_face_encodings)