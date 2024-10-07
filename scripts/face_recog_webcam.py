import face_recognition
import cv2
import numpy as np
import os
import yaml


video_capture = cv2.VideoCapture(0)

global known_face_encodings, known_face_names

SCALING_FACTOR = 1
N_SUBJS_TO_LOAD = 10000


def load_templates_dataset(templates_dir_path):
    global known_face_encodings, known_face_names
    templates_paths = os.listdir(templates_dir_path)
    known_face_encodings = [np.load(os.path.join(templates_dir_path, template_path)) for template_path in templates_paths]
    known_face_names = [template_path.split('.')[0] for template_path in templates_paths]

    final_encodings = known_face_encodings[:N_SUBJS_TO_LOAD]
    final_names = known_face_names[:N_SUBJS_TO_LOAD]

    known_face_encodings = final_encodings
    known_face_names = final_names

    print(f'Total known face encodings: {len(known_face_encodings)}')
    print(f'Total known face names: {len(known_face_names)}')
    #print(f'Known face names: {known_face_names}')
    print('----------')
            

def run_with_face_recognition():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    final_face_names = []
    face_distances = []
    is_processing_frame = True

    frame_number = 0
    while True:
        # Grab a single frame of video
        _, frame = video_capture.read()

        if frame is None:
            print('Could not read frame from video capture')
            break

        # Only process every other frame of video to save time
        if is_processing_frame:
            # Resize frame of video for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=1/SCALING_FACTOR, fy=1/SCALING_FACTOR)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            final_face_names, final_face_dists, final_face_locations = [], [], []
            for face_enc, face_loc in zip(face_encodings, face_locations):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=0.5)  # 0.54
                face_distances = face_recognition.face_distance(known_face_encodings, face_enc)
                name = "Unknown"
                face_dist = -1

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_dist = face_distances[best_match_index]

                final_face_names.append(name)
                final_face_dists.append(face_dist)
                final_face_locations.append(face_loc)
                

        # Display the results
        for (top, right, bottom, left), name, f_dist in zip(final_face_locations, final_face_names, final_face_dists):
            # Scale back up face locations since the frame we detected in was scaled
            top *= SCALING_FACTOR
            right *= SCALING_FACTOR
            bottom *= SCALING_FACTOR
            left *= SCALING_FACTOR

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 30), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, f'Distance: {round(f_dist, 5)}', (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, f'Frame Number: {frame_number}', (40, 40), font, 1.0, (0, 0, 255), 1)

            print(f'face_distances[0]: {f_dist}')

        # Display the resulting image
        cv2.imshow('Video', frame)
        
        # Hit 'q' on the keyboard to quit!
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
            
        frame_number += 1
        is_processing_frame = not is_processing_frame

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))
    
    load_templates_dataset(config['templates_dir'])

    if config['framework'] == 'face_recognition':
        run_with_face_recognition()
    