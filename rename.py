import os

# Takes every file from path/_Unprocessed, renames them to PST00001, PST00002, etc.,
# and moves them into path/destination folder. If there are already files following
# this naming convention in a folder and you want to add more photos following the
# indexing, comment lines 27-38 and uncomment lines 42-50.

path = os.path.expanduser("~/Pictures/")
unprocessedFolder = "_Unprocessed/"

print( "#----------------------------------------------------------#")
print(f"| Please enter dest folder in {path}. Replace '/' with ':' |")
print( "#----------------------------------------------------------#")
destFolder = input("Dest folder: ")
if (destFolder[-1]!="/"): destFolder = destFolder + "/"
try:
    os.remove(path+unprocessedFolder+".DS_Store")
except:
    pass
try:
    os.remove(path+destFolder+".DS_Store")
except:
    pass

### Renumber ###
idx = 1
for file in sorted(os.listdir(path+unprocessedFolder)):
    if os.path.isfile(path+unprocessedFolder+file):
        oldName, fileExt = os.path.splitext(file)
        oldPath = path + unprocessedFolder + oldName + fileExt
        newName = "PST00000"
        newPath = path + destFolder + newName[:-len(str(idx))] + str(idx) + fileExt
        if os.path.isfile(newPath): newPath = newPath + "a"
        lastFile = sorted(os.listdir(path))[-1]
        os.rename(oldPath, newPath)
        idx += 1
print(f"Numbering complete. {idx-1} file(s) processed")

### Add to existing folder ###
# # print(int(os.path.splitext(sorted(list(filter(os.path.isfile, [path+f for f in os.listdir(path)])))[-1])[0][-5:]))
# for file in sorted(os.listdir(path+unprocessedFolder)):
#     if os.path.isfile(path+unprocessedFolder+file):
#         oldName, fileExt = os.path.splitext(file)
#         oldPath = path + unprocessedFolder + oldName + fileExt
#         newName = 'PST00000'
#         nextIdx = int(os.path.splitext(sorted(list(filter(os.path.isfile, [path+f for f in os.listdir(path)])))[-1])[0][-5:]) + 1
#         newPath = path + destFolder + newName[:-len(str(nextIdx))] + str(nextIdx) + fileExt
#         if os.path.isfile(newPath): newPath = newPath + "a"
#         os.rename(oldPath, newPath)

### Resize ###
# for file in os.listdir(path):
#     oldpath = path + file
#     im = Image.open(oldpath)
#     im = im.resize((640,480))
#     im.save(oldpath)