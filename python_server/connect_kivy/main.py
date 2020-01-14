import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera



from kivy.utils import platform
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

Builder.load_string('''
<Login>:
    orientation: 'vertical'
    

    Button:
        text: 'Submit'
        on_press: XinCamera.play = not XinCamera.play
        size_hint_y: None
        height: '48dp'
''')





class Login(BoxLayout):
    #wtf init method is called after build. this should only be used to store button functions. never put init method in this function. dk when is it called
    pass



class LiuHe(App):

    def __init__(self):
        super(LiuHe, self).__init__()

    def build(self):
        layout = Login()
        return layout

    def on_stop(self):
        pass

if __name__ == "__main__":
    LiuHe().run()