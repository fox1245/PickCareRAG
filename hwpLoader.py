from Loader import Loader
import init


class hwpLoader(Loader):
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load(self):
        #보안 모듈을 사용하여 보안 팝업 우회 필요 -> 번거로움
        init.pythoncom.CoInitialize()
        self.hwp = init.win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
        self.hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModuleExample")
        #self.hwp.Open(self.file_path, "HWP", "forceopen:true")
        #"forceopen:1;suspendpassword:1;versionwarning:0"
        self.hwp.Open(self.file_path, "HWP", "forceopen:1;suspendpassword:1;versionwarning:0")
        
        self.hwp.InitScan()
        self.hwp.Run("SelectAll")
        text = self.copy_all_text()
        file_name = str(self.file_path).split('/')[-1]
        file_name = file_name.replace(" ","_").replace("hwp", "txt")
        with open(fr"{file_name}" , "w", encoding = "utf-8") as f:
            f.write(text)
        init.pythoncom.CoUninitialize()
        return text
    
    def load2(self):
        try:
                
            hwp = init.Hwp(visible=False)
            hwp.open(self.file_path,  "HWP", "forceopen:1;suspendpassword:1;versionwarning:0")
            text_result = hwp.GetTextFile("TEXT", "")
            text = text_result
            file_name = str(self.file_path).split('/')[-1]
            file_name = file_name.replace(" ","_").replace("hwp", "txt")
            with open(fr"{file_name}" , "w", encoding = "utf-8") as f:
                f.write(text)
            return text
        except Exception as e:
            print(e)
        finally:
            hwp.quit()
        
    
        
        
        
    def copy_all_text(self):
        text_buf = []
        res = ""
        self.hwp.SetPos(0,0,0)
        while True:
            state, text = self.hwp.GetText()
            res += text
            if state <= 1:
                return res
            
            
            
# if __name__ == "__main__":
    
#     h = hwpLoader(r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp")
#     res = h.load2()
#     print(res)
            
            
    
    
    
    
        
        
        
        