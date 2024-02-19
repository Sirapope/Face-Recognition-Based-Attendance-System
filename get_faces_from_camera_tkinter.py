import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import re
import requests

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0  #  cnt for counting faces in current frame
        self.existing_faces_cnt = 0  # cnt for counting saved faces
        self.ss_cnt = 0  #  cnt for screen shots

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")

        # PLease modify window size here if needed
        self.win.geometry("1000x500")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(1)  # Get video stream from camera

        # self.cap = cv2.VideoCapture("test.mp4")   # Input local video

    #  Delete old face folders
    def GUI_clear_data(self):
        #  "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_get_input_id(self):
        self.input_name_char = self.input_student_id.get()
        if not self.input_name_char:
            self.log_all.config(text="Please enter a student ID", fg="red")
            return

        # Create face folder based on student ID
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)
        # Optionally, you can display a success message
        self.log_all.config(text=f"Face folder for student ID {self.student_id} created successfully!", fg="green")

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                text="Face Register",
                font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        # Add input fields for student ID, first name, last name
        # Input fields for student ID
        tk.Label(self.frame_right_info, text="Student ID: ").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_student_id = tk.Entry(self.frame_right_info)
        self.input_student_id.grid(row=6, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                text='Input Student ID',
                command=self.GUI_get_input_id).grid(row=6, column=2, padx=5)

        tk.Label(self.frame_right_info, text="First Name: ").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_first_name = tk.Entry(self.frame_right_info)
        self.input_first_name.grid(row=7, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Last Name: ").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_last_name = tk.Entry(self.frame_right_info)
        self.input_last_name.grid(row=7, column=1, sticky=tk.W, padx=0, pady=2)

        # Add dropdowns for faculty and field selection
        tk.Label(self.frame_right_info, text="Faculty: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=2)
        self.faculty_options = ["SIT"]  # Modify with actual options
        self.faculty_var = tk.StringVar(self.frame_right_info)
        self.faculty_var.set(self.faculty_options[0])  # Default value
        self.dropdown_faculty = tk.OptionMenu(self.frame_right_info, self.faculty_var, *self.faculty_options)
        self.dropdown_faculty.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Field: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
        self.field_options = ["IT", "CS", "DSI"]  # Modify with actual options
        self.field_var = tk.StringVar(self.frame_right_info)
        self.field_var.set(self.field_options[0])  # Default value
        self.dropdown_field = tk.OptionMenu(self.frame_right_info, self.field_var, *self.field_options)
        self.dropdown_field.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2)

        # Add input fields for email, password, and confirm password
        tk.Label(self.frame_right_info, text="Email: ").grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_email = tk.Entry(self.frame_right_info)
        self.input_email.grid(row=10, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Password: ").grid(row=11, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_password = tk.Entry(self.frame_right_info, show="*")
        self.input_password.grid(row=11, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Confirm Password: ").grid(row=12, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_confirm_password = tk.Entry(self.frame_right_info, show="*")
        self.input_confirm_password.grid(row=12, column=1, sticky=tk.W, padx=0, pady=2)

        # Add a button to trigger registration process
        tk.Button(self.frame_right_info,
                text='Register',
                command=self.register_face).grid(row=13, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        # Show log in GUI
        self.log_all.grid(row=14, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        
                
        self.frame_right_info.pack()



    
    def register_face(self):
        # Get the values entered by the user
        student_id = self.input_student_id.get()
        first_name = self.input_first_name.get()
        last_name = self.input_last_name.get()
        faculty = self.faculty_var.get()
        field = self.field_var.get()
        email = self.input_email.get()
        password = self.input_password.get()
        confirm_password = self.input_confirm_password.get()

        # Validation checks
        if not all([student_id, first_name, last_name, email, password, confirm_password]):
            self.log_all.config(text="Please fill in all fields", fg="red")
            return

        if not self.validate_email(email):
            # Email is valid, proceed with registration
            self.log_all.config(text="Invalid email address!", fg="red")
            return
        
        if password != confirm_password:
            self.log_all.config(text="Passwords do not match", fg="red")
            return

        # Here you can add additional validation or processing logic as needed

        # Once validated, you can proceed with saving the face or performing any other action
        self.save_current_face()

        
        # API endpoint
        api_url = "http://10.4.85.17:8081/api/student"

        # Request payload
        payload = {
            "student_id": student_id,
            "firstname": first_name,
            "lastname": last_name,
            "faculty": faculty,
            "field": field,
            "email": email,
            "password": password
        }

        # Send POST request to the API
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 201:
                self.log_all.config(text="Registration successful!", fg="green")
            elif response.status_code == 400:
                error_message = response.json().get("message", "Registration failed!")
                self.log_all.config(text=error_message, fg="red")
            else:
                self.log_all.config(text="Failed to register. Please try again later.", fg="red")
        except Exception as e:
            print("Error:", e)
            self.log_all.config(text="Failed to connect to the server. Please try again later.", fg="red")

        # Optionally, you can display a success message
        self.log_all.config(text="Face registered successfully!", fg="green")

    def validate_email(self, email):
        # Regular expression pattern for email validation
        pattern = r'^[\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+$'
        return re.match(pattern, email)

    
    def GUI_info2(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info,
                  text='Clear',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='Input',
                  command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Save face image").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)

        # Show log in GUI
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)

        # Start from person_1
        else:
            self.existing_faces_cnt = 0

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        #  Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt) + "_" + \
                                    self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  #  Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    #  Create blank image according to the size of face detected
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"
        else:
            self.log_all["text"] = "Please run step 2!"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #  Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            #  Face detected
            if len(faces) != 0:
                #   Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    #  Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
