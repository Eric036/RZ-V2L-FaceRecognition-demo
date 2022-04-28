OPENCV_LINK = -isystem ${SDKTARGETSYSROOT}/usr/include/opencv4 \
			  -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_dnn \

EIGEN = -I ./eigen3 \

#OPENCV_CONTRIB = -I /home/eric/Desktop/opencv_contrib/opencv-4.1.0/opencv_arm/include/opencv4/ \

BSP_080_SDK_FLAG = \
			  -ljpeg -lwebp -ltiff -lz -ltbb -lgtk-3 -lpng16 -lgdk-3 -lcairo  \
			  -llzma -lrt -lcairo-gobject \
			  -lxkbcommon -lwayland-cursor -lwayland-egl -lwayland-client -lepoxy \
			  -lfribidi -lharfbuzz -lfontconfig \
			  -lglib-2.0 -lgobject-2.0 -lgdk_pixbuf-2.0 -lgmodule-2.0 -lpangocairo-1.0 \
			  -latk-1.0 -lgio-2.0 -lpango-1.0 -lfreetype -lpixman-1 -luuid -lpcre \
			  -lmount -lresolv -lexpat -lpangoft2-1.0 -lblkid \


all: sample_app_faceRecognition_cam.cpp
	${CXX} -std=c++14 mtcnn.cpp camera.cpp image.cpp wayland.cpp sample_app_faceRecognition_cam.cpp \
	-lwayland-client \
	${OPENCV_LINK} ${BSP_080_SDK_FLAG} ${EIGEN} \
	-lpthread -O2 -ldl ${LDFLAGS} -o sample_app_faceRecognition_cam

clean:
	rm -rf *.o sample_app_faceRecognition_cam
