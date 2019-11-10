import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
 
 
class FirstButtonApp(App):
    def build(self):
        button =  Button(text="START")
        button.bind(on_press = self.click) 
        return button
    
    def click(self, events):
        import os
        print("Clicked")
        import temp
        result = temp.get_dimension()
        # result = "Test"
        os.system("python result.py {}".format(result))
    
 
if __name__ == "__main__":
   FirstButtonApp().run()