# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

DEPTHANYTHING_DIR = /Users/adithyapalle/work/CS5330/video_special_effects/depthAnything
ONNX_RUNTIME_DIR = $(DEPTHANYTHING_DIR)/onnxruntime-osx-arm64-1.20.1

# OSX include paths for MacPorts
CFLAGS = -Wc++11-extensions -std=c++14 -I/usr/local/opt/opencv/include/opencv4 -I/usr/local/include -I$(ONNX_RUNTIME_DIR)/include  -DENABLE_PRECOMPILED_HEADERS=OFF -Wc++14-extensions

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to wwhere your openCV libraries are stored
# these settings work for macports
LDFLAGS = -L/usr/local/opt/opencv/lib -L/usr/local/lib -L$(ONNX_RUNTIME_DIR)/lib # opencv libraries are here

# opencv libraries and MacOS frameworks
# these settings work for macOS and macports
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime


# Location to write the executables
BINDIR = .

# rules for making various executables
# mnist: mnist-cv.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# depth: depthanything.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
# 	install_name_tool -add_rpath /opt/local/lib ./$@

da2: da2-example.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /usr/local/lib ./$@

da2vid: da2-video.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /usr/local/lib ./$@

clean:
	rm -f *.o *~ 
