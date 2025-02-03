import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

# import scipy.linalgrrrrr
import sympy
import math

def get_rotation_matrix(u, a): # u: axis ; a: angle
    return [
        [(u[0]*u[0]) * (1-np.cos(a)) +      np.cos(a), (u[1]*u[0]) * (1-np.cos(a)) - u[2]*np.sin(a),\
              (u[2]*u[0]) * (1-np.cos(a)) + u[1]*np.sin(a)],
        [(u[0]*u[1]) * (1-np.cos(a)) + u[2]*np.sin(a), (u[1]*u[1]) * (1-np.cos(a)) +      np.cos(a),\
             (u[2]*u[1]) * (1-np.cos(a)) - u[0]*np.sin(a)],
        [(u[0]*u[2]) * (1-np.cos(a)) - u[1]*np.sin(a), (u[1]*u[2]) * (1-np.cos(a)) + u[0]*np.sin(a),\
             (u[2]*u[2]) * (1-np.cos(a)) +      np.cos(a)]
                 ]
                 
class SAE:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        jevois.LINFO("PythonTest Constructor")
        self.frame = 0
        self.timer = jevois.Timer("pytest", 100, jevois.LOG_INFO)
        
        self.ASPECT_RATIO = 3 / 4
        self.H_FOV = 65 * math.pi / 180
        self.V_FOV = 2 * math.atan(math.tan(self.H_FOV / 2) * self.ASPECT_RATIO)
        jevois.LINFO("    V_FOV: " + str(self.V_FOV))
        self.counter = 0
        self.index = 0
        self.t = 0
        
    # #################################################################
    def init(self):
        jevois.LINFO("PythonTest JeVois init")
        
        pc_misc = jevois.ParameterCategory("Miscellaneous Parameters", "")
        self.gray_threshold = jevois.Parameter(self, "Gray Low Threshold","int",\
            "Low Threshold for countours finding", 10, pc_misc)
        self.min_area_threshold = jevois.Parameter(self, "Minimal Shape Area","int",\
            "Minimal shape area recognized", 500, pc_misc)
        self.min_centroid_diff = jevois.Parameter(self, "Minimal Centroid Distance","int",\
            "Minimal distance between centroids", 200, pc_misc)
            
        pc_camera = jevois.ParameterCategory("Camera Parameter", "")
        self.camera_posx = jevois.Parameter(self, "Camera Position X (m)","float",\
            "Camera Position X (m)", 0.0, pc_camera)
        self.camera_posy = jevois.Parameter(self, "Camera Position Y (m)","float",\
            "Camera Position Y (m)", 0.0, pc_camera)
        self.camera_posz = jevois.Parameter(self, "Camera Position Z (m)","float",\
            "Camera Position Z (m)", 0.21, pc_camera)
            
        self.camera_pitch_deg = jevois.Parameter(self, "Camera Pitch (°)","float",\
            "Camera Pitch (°)", 30, pc_camera)
            
            
        self.camera_pos = np.array([self.camera_posx.get(), self.camera_posy.get(), self.camera_posz.get()])
        self.camera_pitch = self.camera_pitch_deg.get() * math.pi / 180
        
        self.x_axis = np.array([1, 0, 0])
        self.y_axis = np.array([0, 1, 0])
        self.z_axis = np.array([0, 0, 1])
        
        self.camera_rotation_M = get_rotation_matrix(self.y_axis, self.camera_pitch)
        
        self.camera_direction = np.dot(self.camera_rotation_M, np.array([1, 0, 0]))
        self.camera_direction = self.camera_direction / np.linalg.norm(self.camera_direction)
        
        self.camera_right = self.y_axis # Just because we rotate only on pitch
        self.camera_up = np.cross(self.camera_direction, self.camera_right)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)
        
    def findTargets(self, img, display=False) :
        # Conversion en HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Détection des objets rouges
        # Masque inférieur (0-10)
        lower_red1 = np.array([0, 200, 100]) #S = 120
        upper_red1 = np.array([2, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        # Masque inférieur (0-10)
        lower_red2 = np.array([170, 200, 100]) #S = 120
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        # Combinaison des deux masques
        mask = mask1 + mask2
        
        # Trouver les contours des objets détectés
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcul de l'aire minimale dynamique
        # min_area_threshold = 0.001 * (img.shape[0] * img.shape[1])
        min_area_threshold = self.min_area_threshold.get()
        
        # Dimensions de l'image
        h, w = img.shape[:2]
        
        # Calcul du centre de l'image
        center_x_img = w // 2
        center_y_img = h // 2
        
        # Déterminer un facteur d'échelle
        scale_factor = min(w,h) / 1000
        font_scale = 1 * scale_factor
        font_thickness = int(2*scale_factor)
        
        # Copier l'image 
        output_img = img.copy()
        
        # Liste pour stocker les positions des objets
        detected_objects = []
        
        # Boucle sur les contours détectés
        for contour in contours:
            if cv2.contourArea(contour) > min_area_threshold:
                x, y, w_obj, h_obj = cv2.boundingRect(contour)
                
                rect_thickness = int(3*scale_factor)
                cv2.rectangle(output_img, (x, y), (x+w_obj, y+h_obj), (0, 255, 0), rect_thickness)
                
                # Coordonnées du centre de l'objet par rapport au centre de l'image
                base_center_x = (x + w_obj // 2) - center_x_img
                base_center_y = -(y + h_obj - center_y_img)
                base_center_coordinates_scaled = (base_center_x, base_center_y)
                base_center_coordinates = np.array((x+w_obj//2, y+h_obj))
                
                
                cv2.circle(output_img, base_center_coordinates, int(5*scale_factor), (255, 0, 0), -1)
                #cv2.putText(output_img, f"{base_center_coordinates_scaled}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                
                detected_objects.append(base_center_coordinates)
                
        return output_img, detected_objects
    
    def findTargetsRealPosition(self, centroids, img_width, img_height):
        image_center = np.array([img_width / 2, img_height / 2])
        focal_ref = 772.692
        
        table_position = []
        # ------------------
        # Compute centroids 3D position
        for c in centroids:
            centered_point = np.array((c[0], img_height - c[1])) - image_center
            
            dist_alpha = np.linalg.norm(centered_point)
            
            alpha = np.arctan(dist_alpha / focal_ref)
            theta = math.pi - np.arccos(centered_point[0] / dist_alpha)
            if centered_point[1] < 0:
                theta = -theta
            
            
            rotation_Mroll = get_rotation_matrix(self.camera_direction, theta)
            rotation_Myaw = get_rotation_matrix(self.camera_up, alpha)
            
            obj_direction = np.dot(rotation_Myaw, self.camera_direction)
            obj_direction = np.dot(rotation_Mroll, obj_direction)
            
            step = -self.camera_pos[2] / obj_direction[2]
            y = self.camera_pos[1] + obj_direction[1] * step
            x = self.camera_pos[0] + obj_direction[0] * step
            centroid_relative_tpos = np.array([x, y])
            
            table_position.append(centroid_relative_tpos)
        
        return table_position
        
    def findClosestObject(self, centroids, table_pos, img):
        obj_dist = 100000
        x_obj = 0
        y_obj = 0
        x_img = 0
        y_img = 0
        for c, p in zip(centroids, table_pos):
            cv2.putText(img, "3D x:" + str(round(p[0], 3)) + " ; y:" + str(round(p[1], 3)), \
                (int(c[0] - 50), int(c[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            x = int(round(p[0], 3)*1000)
            y = int(round(p[1], 3)*1000)
            obj_dist_new = np.sqrt(x * x + y * y)
            cv2.putText(img, f"{obj_dist_new}", (int(c[0]), int(c[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            if obj_dist_new < obj_dist and obj_dist_new > 100:
                obj_dist = obj_dist_new
                x_img = int(c[0])
                y_img = int(c[1])
                x_obj = x
                y_obj = y
         
        cv2.circle(img, (x_img, y_img), 5, (255, 255, 0), -1)
                
        return x_obj, y_obj
            
        
    def encodeAndSendPosition(self, msg_function, msg_payload_length, msg_payload) :

        message = b'\xfe'
        message += msg_function.to_bytes(2, "big")
        message += msg_payload_length.to_bytes(2, "big")
        message += msg_payload
        message += self.calculateChecksum(msg_function, msg_payload_length, msg_payload).to_bytes(1, "big")
        
        return message
        
    def calculateChecksum(self, msg_function, msg_payload_length, msg_payload) :
        checksum = 0
        checksum ^= 0xFE
        checksum ^= msg_function.to_bytes(2, "big")[0]
        checksum ^= msg_function.to_bytes(2, "big")[1]
        checksum ^= msg_payload_length.to_bytes(2, "big")[0]
        checksum ^= msg_payload_length.to_bytes(2, "big")[1]
        
        for p in msg_payload:
            checksum ^= p
            
        return checksum

    # ###################################################################################################
    ## Process function with GUI output (JeVois-Pro mode):
    def processGUI(self, inframe, helper):
        idle, winw, winh = helper.startFrame()
        
        self.timer.start()
        # ---------------------------------
        
        img, centroids = self.findTargets(inframe.getCvRGB(), display=True)
        
        table_pos = self.findTargetsRealPosition(centroids, img.shape[1], img.shape[0])
        
        x_closest_obj, y_closest_obj = self.findClosestObject(centroids, table_pos, img)
        
        if table_pos :
            if y_closest_obj < 0 :
                y_closest_obj = -y_closest_obj
                sign = True
            else :
                sign = False
                
            if x_closest_obj >= 0:
                dataToSend = x_closest_obj.to_bytes(4, "big")
                dataToSend += y_closest_obj.to_bytes(4, "big")
                if self.t < 5 : 
                    self.t+=1
                else : 
                    self.t=0
                    if sign :
                        jevois.sendSerial(self.encodeAndSendPosition(0x0099, len(dataToSend), dataToSend))
                    else : 
                        jevois.sendSerial(self.encodeAndSendPosition(0x0098, len(dataToSend), dataToSend))
                    
        # Draw Centroid and table positions
        #    cv2.circle(img, (int(c[0]), int(c[1])), 5, (255, 0, 0), -1)
        #    cv2.putText(img, "3D x:" + str(round(p[0], 3)) + " ; y:" + str(round(p[1], 3)), \
        #        (int(c[0] - 50), int(c[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #    
        #    p[1]=p[1] if p[1]>0 else -p[1]
        #    if p[0] >= 0:
        #        dataToSend = int(round(p[0], 3)*1000).to_bytes(4, "big")
        #        dataToSend += int(round(p[1], 3)*1000).to_bytes(4, "big")
        #        if round(p[1], 3) >= 0 :
        #            jevois.sendSerial(self.encodeAndSendPosition(0x0098, len(dataToSend), dataToSend))
        #        else : 
        #            jevois.sendSerial(self.encodeAndSendPosition(0x0099, len(dataToSend), dataToSend))
            #jevois.sendSerial(f"{x};{y}")
        helper.drawImage("img", img, True, 0, 0, 0, 0,False, False)
        # ---------------------------------
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh)
        helper.endFrame()
        
        
        














































