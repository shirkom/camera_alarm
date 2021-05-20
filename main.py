from typing import List
from pathlib import Path
import face_recognition
import cv2
import os
import glob


# write the path of dir with image of known people
# Please add a system environment variable pointing to the known faces directory path
PATH = os.environ['KNOWN_FACES_PATH']


def parse_known_dir(path_to_directory: str):
    """
    :param path_to_directory: path to dir with all the images of known people (name of the file is name of person)
    :return: list of names and list of encoding images
    """
    list_of_files = glob.glob(path_to_directory + "/*.*")
    list_of_names = []
    list_of_encoding = []
    for path in list_of_files:
        list_of_names.append(Path(path).stem)
        known_image = face_recognition.load_image_file(path)
        list_of_encoding.append(face_recognition.face_encodings(known_image)[0])

    return list_of_names, list_of_encoding


NAMES, ENCODING = parse_known_dir(PATH)


def parse_result(result: List[bool]) -> str:
    """
    :param result: List of booleans result[i] is true if NAMES[i] is the person
    :return: The names of persons
    """
    i = 0
    for item in result:
        if item:
            return NAMES[i]
        i += 1

    return "unknown"


def identify_faces_in_picture(unknown_image) -> List[tuple]:
    """
    :param unknown_image - An image unknown_image
    :return: A list of tuples contains name and location.
    """
    unknown_image = cv2.resize(unknown_image, (0, 0), fx=0.25, fy=0.25)
    ans: list = []
    unknown_encodings_found = face_recognition.face_encodings(unknown_image)
    unknown_location_found = face_recognition.face_locations(unknown_image)

    for unknown_encoding, unknown_location in zip(unknown_encodings_found, unknown_location_found):
        result = face_recognition.compare_faces(ENCODING, unknown_encoding)
        ans.append((parse_result(result), (unknown_location[0] * 4, unknown_location[1] * 4, unknown_location[2] * 4,
                                           unknown_location[3] * 4)))

    return ans


def draw_string_on_image(image, text: str, text_point: tuple):
    """
    :param image:
    :param text: a string - name of the person
    :param text_point:
    :return: the image with the name
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = text_point
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def calculate_text_point(left_up_corner: tuple):
    """
    :param left_up_corner: a tuple of face location
    :return: text_point: which pointer the text will write
    """
    text_point = (left_up_corner[0], left_up_corner[1] - 4)
    return text_point


def draw_square_on_face(cv2_image, text: str, upper_left_corner: tuple, lower_right_corner: tuple):
    """
    :param cv2_image
    :param text: a string name
    :param upper_left_corner: a tuple
    :param lower_right_corner: a tuple
    :return: the image with square on face location
    """
    text_point = calculate_text_point(upper_left_corner)
    color = (255, 0, 0)
    thickness = 1
    image = cv2.rectangle(cv2_image, upper_left_corner, lower_right_corner, color, thickness)
    draw_string_on_image(image, text, text_point)
    return image


def beep():
    """
    make a beep sound
    """
    for i in range(0, 6):
        duration_seconds = 0.1
        freq_hz = 245
        os.system('play -nq -t alsa synth {} sine {}'.format(duration_seconds, freq_hz))


def capture_video_from_camera():
    """
    Capture image from camera and identify which faces are in it, if recognizes an unknown face make noise.
    press "q" to stop.
    :return: none
    """
    cap = cv2.VideoCapture(0)
    face_list = []
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        face_recognition_img = frame[:, :, ::-1]
        if i % 5 == 0:  # every 5 frames
            face_list = identify_faces_in_picture(face_recognition_img)
        for name, (p1, p2, p3, p4) in face_list:
            left_up_corner = (p4, p1)
            right_down_corner = (p2, p3)
            frame = draw_square_on_face(frame, name, left_up_corner, right_down_corner)
            if name == "unknown":
                beep()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_video_from_camera()

