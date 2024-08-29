#%%
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import Sequential, load_model
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

cap = cv2.VideoCapture(0) 
model = None
class_names = None


def showimg(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


def resize(img):
    target_size = (28, 28)

    # Get the original dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Taller than wide or square
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding to be added
    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top
    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left

    # Add padding to the resized image to make it 28x28
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def symbol_position(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('blacky', binary)
    # showimg(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold (adjust this value based on your needs)
    min_area = 5

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    symbols_loc = [cv2.boundingRect(contour) for contour in filtered_contours]
    symbols_loc.sort(key= lambda x : x[0] )
    return symbols_loc,binary



#%%
def vedioTracking():
    while True:
        _, frame = cap.read() 

        # Flip image 
        # frame = cv2.flip(frame, 1) 
        # Draw a rectangle on the frame
        x, y, w, h = 200, 200, 200, 100  # Example values (x, y) is top-left and (w, h) is width and height
        cropped_frame = frame[y:y+h, x:x+w].copy()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        # Process the image to get the position and binary mask
        pos, binary = symbol_position(cropped_frame)

        # Display text at the specified position
        if pos:
            x_pos, y_pos,w_pos,h_pos = pos[-1]  # Assuming `pos` contains the (x, y) position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 0)  # White color
            thickness = 2
            try:
                symbols = [resize(binary[y:y+h, x:x+w]) for x, y, w, h in pos]
            except:
                pass
            result=photo_add(symbols)
            if result:
                text = "= "+str(result)
                # Draw the text on the image
                cv2.putText(frame, text, (x_pos+x+w_pos+20, y_pos+y+h_pos), font, font_scale, color, thickness)
            for p in pos:
                conx,cony,conW,conH = p
                cv2.rectangle(frame,(conx+x,cony+y),(conx+x+conW,cony+y+conH),(255, 0, 0),1)

        # Optionally, display the frame with the text (e.g., in a window)
        cv2.imshow('Frame with Text', frame)
        if cv2.waitKey(1) & 0xff == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()

#%%


# Load the model and class names once globally

def load_images_from_folder(folder_path):
    images = []
    labels = []

    for image_file in os.listdir(folder_path):
        try:
            label, _ = image_file.split('-')
            labels.append(label)

            img_path = os.path.join(folder_path, image_file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale

            img_array = np.array(img)
            _, binary_image = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY_INV)

            images.append(binary_image)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    return np.array(images), numeric_labels, label_encoder.classes_

def train_model(dataset_path='symbols'):
    print('Please wait, the network is training...!')
    images, labels, class_names = load_images_from_folder(dataset_path)
    print("first lable = ",labels)
    images = images.reshape((images.shape[0], 28, 28)).astype('float32') / 255  # Flatten and normalize
    labels = to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

    model.save('symbol_model.h5')  # Save the trained model
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    f.close()
    with open('class_names.pkl', 'rb') as file:
            loaded_classes = pickle.load(file)
    file.close()
    print("loaged class ========== " )
    print(loaded_classes)

    print("Model trained and saved as 'symbol_model.h5'")
    print(f"Class names: {class_names}")  # Save this for later use in prediction
    return model, class_names

def predict_image(image_list, model, class_names):
    img = (image_list)
    # img = img.resize((28, 28,1))  # Resize to 28x28 pixels
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255 
    # img_array = np.array(img).reshape(1, 784).astype('float32') / 255
    predictions = model.predict(img)
    prediction = np.argmax(predictions)
    confidence = predictions[0][prediction]  # Confidence score



    return class_names[prediction], confidence
def load_trained_model():
    # try:
        model = load_model('symbol_model.h5')
        with open('class_names.pkl', 'rb') as file:
            loaded_classes = pickle.load(file)

        print("Loaded saved model.")
        return model, loaded_classes
    # except Exception as e:
        # print(f"Failed to load model: {e}")
        # return None, None
def photo_add(list):
    flag_num1=True
    flag_num2=False
    fistnum=0
    lastnum=0
    opt=None
    result=None
    for i in list:
        label, confidence = predict_image(i, model, class_names)
        if (confidence*100)>=30:
                if label.isdigit() and flag_num1:
                    fistnum= fistnum * 10 + int(label)
                elif label.isalpha() and flag_num1:
                    flag_num1=False
                    opt=label
                elif label.isdigit():
                    flag_num2=True
                    lastnum=lastnum * 10 + int(label)
        else:
            print("Prediction failed. Confidence is too low.")
            print(f"Predicted Label: {label}, Confidence: {confidence:.2f}")
    if  flag_num2 :   
        if opt=='plus':
            result=fistnum+lastnum
        elif opt=='minus':
            result=fistnum-lastnum
        elif opt == "slash":
            if lastnum != 0:
                result=fistnum/lastnum
            else:
                result="Division by 0"
        else:
            result=fistnum*lastnum
    return result


# Main Execution Flow
if __name__ == "__main__":

#=======================my code start ====================================
    # vedioTracking()
    # model, class_names = load_trained_model()
    # image = cv2.imread(r'C:\Mathlab\my code\ANN\annpro\ann_project\imggg.jpg')
    # pos,binary=symbol_position(image)
    # symbols = [resize(binary[y:y+h, x:x+w]) for x, y, w, h in pos]
    # print("i am hear")
    # text=photo_add(symbols)
    # print(text)
    # count=1
    # for sym in symbols:
    #     showimg(sym)
    #     count+=1
#================== my code end =============================================
    # Step 1: Train the model (only once)
    # Check if the model is already trained and saved
    if os.path.exists('symbol_model.h5'):
        model, class_names = load_trained_model()
        print(class_names)
    else:
        # Step 1: Train the model (only if not already trained)
        model, class_names = train_model()

    # Step 2: Predict using the saved model
    print('Your code is predicting. Please be cool...!')
    vedioTracking()
