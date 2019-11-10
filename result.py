from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

data = ""


class FirstKivy(App):
    def build(self):
        return Label(text=data)

if __name__ == "__main__":
    import sys
    data = str(sys.argv[1])
    FirstKivy().run()
    
 