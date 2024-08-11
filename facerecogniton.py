import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
from PIL import Image, ImageTk
import numpy as np

# Function to check if the user ID already exists
def check_id_exists(user_id):
    if not os.path.exists("user_info.txt"):
        return False
    with open("user_info.txt", "r") as file:
        for line in file:
            info = line.strip().split(",")
            if info[0] == str(user_id):
                return True
    return False

# Function to create the popup for generating dataset
def generate_dataset_popup():
    popup = tk.Toplevel(window)
    popup.title("Generate Dataset")
    popup.geometry("400x300")
    popup.configure(bg="#a1c4fd")

    def on_closing():
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_closing)

    l1 = tk.Label(popup, text="Name", font=("Arial", 14), bg="#a1c4fd")
    l1.grid(column=0, row=0, pady=10, padx=10)
    name_entry = tk.Entry(popup, width=30, bd=5)
    name_entry.grid(column=1, row=0, pady=10, padx=10)

    l2 = tk.Label(popup, text="ID", font=("Arial", 14), bg="#a1c4fd")
    l2.grid(column=0, row=1, pady=10, padx=10)
    id_entry = tk.Entry(popup, width=30, bd=5)
    id_entry.grid(column=1, row=1, pady=10, padx=10)

    l3 = tk.Label(popup, text="Age", font=("Arial", 14), bg="#a1c4fd")
    l3.grid(column=0, row=2, pady=10, padx=10)
    age_entry = tk.Entry(popup, width=30, bd=5)
    age_entry.grid(column=1, row=2, pady=10, padx=10)

    l4 = tk.Label(popup, text="Address", font=("Arial", 14), bg="#a1c4fd")
    l4.grid(column=0, row=3, pady=10, padx=10)
    address_entry = tk.Entry(popup, width=30, bd=5)
    address_entry.grid(column=1, row=3, pady=10, padx=10)

    def generate_dataset():
        name = name_entry.get()
        user_id = id_entry.get()
        age = age_entry.get()
        address = address_entry.get()

        if name == "" or user_id == "" or age == "" or address == "":
            messagebox.showinfo('Result', 'Please provide complete details of the user')
            return
        if check_id_exists(user_id):
            messagebox.showinfo('Result', 'ID already exists. Please use a different ID.')
            return

        # Save user information to user_info.txt
        user_info = f"{user_id},{name},{age},{address}\n"
        with open("user_info.txt", "a") as file:
            file.write(user_info)

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if face_classifier.empty():
            messagebox.showinfo('Result', 'Error loading face classifier XML file.')
            return

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if not os.path.exists("data"):
                    os.makedirs("data")
                file_name_path = "data/user." + str(user_id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or int(img_id) == 1000:  
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!')
        popup.destroy()

    b1 = tk.Button(popup, text="Generate Dataset", font=("Arial", 14), bg="#1c92d2", fg="white", command=generate_dataset)
    b1.grid(column=1, row=4, pady=20)

# Function to train the classifier
def train_classifier():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo("Result", "Training dataset completed")

# Function to detect faces
def detect_face():
    def get_user_info(user_id):
        with open("user_info.txt", "r") as file:
            for line in file:
                info = line.strip().split(",")
                if info[0] == str(user_id):
                    return info[1], info[2], info[3]
        return None, None, None

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            id, pred = clf.predict(gray_img[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 75:
                name, age, address = get_user_info(id)
                cv2.putText(img, f"Name: {name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                cv2.putText(img, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                cv2.putText(img, f"Address: {address}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        return img

    # Loading classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        if not ret:
            break

        img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to view the dataset
def view_dataset():
    if not os.path.exists("user_info.txt"):
        messagebox.showinfo("Result", "No dataset found")
        return

    dataset_window = tk.Toplevel(window)
    dataset_window.title("View Dataset")
    dataset_window.geometry("600x400")
    dataset_window.configure(bg="#a1c4fd")

    text = tk.Text(dataset_window, font=("Arial", 12), wrap=tk.WORD)
    text.pack(expand=1, fill=tk.BOTH)

    with open("user_info.txt", "r") as file:
        for line in file:
            text.insert(tk.END, line)

# Function to view dataset images
def view_dataset_images():
    data_dir = "data"
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]

    if not image_files:
        messagebox.showinfo("Result", "No images found in dataset")
        return

    images_window = tk.Toplevel(window)
    images_window.title("View Dataset Images")
    images_window.geometry("800x600")
    images_window.configure(bg="#a1c4fd")

    def display_image(index):
        img_path = image_files[index]
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        img_label.config(image=img)
        img_label.image = img

        filename_label.config(text=os.path.basename(img_path))

    img_label = tk.Label(images_window, bg="#a1c4fd")
    img_label.pack(pady=20)

    filename_label = tk.Label(images_window, font=("Arial", 12), bg="#a1c4fd")
    filename_label.pack()

    button_frame = tk.Frame(images_window, bg="#a1c4fd")
    button_frame.pack(pady=20)

    current_img_index = [0]

    def show_prev_image():
        if current_img_index[0] > 0:
            current_img_index[0] -= 1
            display_image(current_img_index[0])

    def show_next_image():
        if current_img_index[0] < len(image_files) - 1:
            current_img_index[0] += 1
            display_image(current_img_index[0])

    prev_button = tk.Button(button_frame, text="<< Previous", font=("Arial", 14), bg="#1c92d2", fg="white", command=show_prev_image)
    prev_button.pack(side=tk.LEFT, padx=20)

    next_button = tk.Button(button_frame, text="Next >>", font=("Arial", 14), bg="#1c92d2", fg="white", command=show_next_image)
    next_button.pack(side=tk.RIGHT, padx=20)

    display_image(current_img_index[0])

# Function to delete a user
def delete_user():
    user_id = simpledialog.askstring("Delete User", "Enter User ID to delete:")
    if not user_id:
        return

    if not check_id_exists(user_id):
        messagebox.showinfo("Result", "User ID not found")
        return

    # Remove user data from user_info.txt
    with open("user_info.txt", "r") as file:
        lines = file.readlines()
    with open("user_info.txt", "w") as file:
        for line in lines:
            if line.split(",")[0] != user_id:
                file.write(line)

    # Remove user images from the dataset
    data_dir = "data"
    for file_name in os.listdir(data_dir):
        if file_name.startswith(f"user.{user_id}."):
            os.remove(os.path.join(data_dir, file_name))

    messagebox.showinfo("Result", "User deleted successfully")

# Function to update user information
def update_user_info():
    user_id = simpledialog.askstring("Update User", "Enter User ID to update:")
    if not user_id:
        return

    if not check_id_exists(user_id):
        messagebox.showinfo("Result", "User ID not found")
        return

    name = simpledialog.askstring("Update User", "Enter new name:")
    age = simpledialog.askstring("Update User", "Enter new age:")
    address = simpledialog.askstring("Update User", "Enter new address:")

    if not name or not age or not address:
        messagebox.showinfo("Result", "Incomplete details provided")
        return

    # Update user info in user_info.txt
    with open("user_info.txt", "r") as file:
        lines = file.readlines()
    with open("user_info.txt", "w") as file:
        for line in lines:
            if line.split(",")[0] == user_id:
                file.write(f"{user_id},{name},{age},{address}\n")
            else:
                file.write(line)

    messagebox.showinfo("Result", "User information updated successfully")

# Function to handle admin login
def admin_login():
    def validate_admin():
        username = username_entry.get()
        password = password_entry.get()

        if username == "admin" and password == "jithendra11":
            login_window.destroy()
            admin_window()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    login_window = tk.Toplevel(window)
    login_window.title("Admin Login")
    login_window.geometry("400x200")
    login_window.configure(bg="#a1c4fd")

    tk.Label(login_window, text="Username:", font=("Arial", 14), bg="#a1c4fd").grid(row=0, column=0, pady=10, padx=10)
    username_entry = tk.Entry(login_window, width=30, bd=5)
    username_entry.grid(row=0, column=1, pady=10, padx=10)

    tk.Label(login_window, text="Password:", font=("Arial", 14), bg="#a1c4fd").grid(row=1, column=0, pady=10, padx=10)
    password_entry = tk.Entry(login_window, width=30, bd=5, show="*")
    password_entry.grid(row=1, column=1, pady=10, padx=10)

    tk.Button(login_window, text="Login", font=("Arial", 14), bg="#1c92d2", fg="white", command=validate_admin).grid(row=2, column=1, pady=20)

# Function to open the administrator window
def admin_window():
    admin_win = tk.Toplevel(window)
    admin_win.title("Administrator")
    admin_win.geometry("1000x500")
    admin_win.configure(bg="#a1c4fd")

    bg_label = tk.Label(admin_win, image=bg_image)
    bg_label.place(relwidth=1, relheight=1)

    buttons = [
        ("Generate Dataset", generate_dataset_popup),
        ("Train Classifier", train_classifier),
        ("View Dataset", view_dataset),
        ("View Dataset Images", view_dataset_images),
        ("Delete User", delete_user),
        ("Update User Info", update_user_info),
        ("Logout", admin_win.destroy)
    ]

    for i, (text, command) in enumerate(buttons):
        b = tk.Button(admin_win, text=text, **button_style, command=command)
        b.place(relx=0.5, rely=0.2 + i * 0.1, anchor=tk.CENTER)

# Function to handle user login
def user_login():
    user_win = tk.Toplevel(window)
    user_win.title("User")
    user_win.geometry("1000x500")
    user_win.configure(bg="#a1c4fd")

    bg_label = tk.Label(user_win, image=bg_image)
    bg_label.place(relwidth=1, relheight=1)

    b3 = tk.Button(user_win, text="Detect Face", **button_style, command=detect_face)
    b3.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b4 = tk.Button(user_win, text="Logout", **button_style, command=user_win.destroy)
    b4.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

# Create the main window
window = tk.Tk()
window.title("Face Recognition System")
window.geometry("1000x500")
window.configure(bg="#a1c4fd")

# Set up a background image
bg_image = Image.open("images/background1.jpg")
bg_image = bg_image.resize((1000, 500), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(window, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Button styles
button_style = {
    "font": ("Arial", 14),
    "bg": "black",  # Background color is black
    "fg": "white",  # Text color is white
    "bd": 3,
    "relief": tk.RAISED
}

# Login as Administrator button
admin_login_button = tk.Button(window, text="Login as Administrator", **button_style, command=admin_login)
admin_login_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

# Login as User button
user_login_button = tk.Button(window, text="Login as User", **button_style, command=user_login)
user_login_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

# Run the main loop
window.mainloop()

