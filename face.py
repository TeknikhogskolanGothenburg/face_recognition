import face_recognition
import os
import cv2
import pickle


KNOWN_DIR = 'known'
UNKNOWN_DIR = 'unknown'
TOLERANCE = 0.55
MODEL = 'cnn'
FRAME_THICKNESS = 3
FONT_THICKNESS = 2


def train():
    known_faces = []
    known_names = []
    for name in os.listdir(KNOWN_DIR):
        print(f'Loading known faces for {name}')
        for filename in os.listdir(f'{KNOWN_DIR}/{name}'):
            print(f'Processing file {filename}')
            path = f'{KNOWN_DIR}/{name}/{filename}'
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            print(f'Found {len(encodings)} face(s) in image {filename}')
            if len(encodings) >= 1:
                encoding = encodings[0]
            else:
                print(f"Did not find any faces in image {filename}")
                continue

            known_faces.append(encoding)
            known_names.append(name)
    return known_faces, known_names


def predict(known_faces, known_names):
    print(f'Processing unknown faces...')
    for filename in os.listdir(UNKNOWN_DIR):
        image = face_recognition.load_image_file(f'{UNKNOWN_DIR}/{filename}')
        locations = face_recognition.face_locations(image)
        # locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        match = False
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            if any(results):
                match = True
                pos = results.index(True)
                name = known_names[pos]

                # Coordinates of face
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = (0, 255, 0)
                # Rectangle for face
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Coordinates for text rectangle
                top_left = (face_location[1] - 100, face_location[2])
                bottom_right = (face_location[1], face_location[2] + 25)

                # Rectangle for text
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                cv2.putText(image, name, (face_location[1] - 90, face_location[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), FONT_THICKNESS)

        if match:
            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)


def save(list_to_save, filename):
    with open(filename, 'wb') as f:
        pickle.dump(list_to_save, f)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    known_faces, known_names = train()
    #save(known_faces, 'faces.pkl')
    #save(known_names, 'names.pkl')
    #known_faces = load('faces.pkl')
    #known_names = load('names.pkl')
    predict(known_faces, known_names)


if __name__ == '__main__':
    main()
