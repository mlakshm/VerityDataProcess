print("start")
exec(open("VerityDataPull_V5.py", encoding="utf8").read())
print("Data Pull completed")
print("starting with Model score process")
exec(open("Verity_Model_Score.py", encoding="utf8").read())
print("end of processing")