import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.util.*;

import net.sourceforge.tess4j.Tesseract;

public class main {
	
	public static void removeVerticalLines(Mat img, int limit) {  
	    Mat lines=new Mat();
	    int threshold = 100;
	    int minLength = 80;
	    int lineGap = 5;
	    int rho = 1;
	    
	    Imgproc.HoughLinesP(img, lines, rho, Math.PI/180, threshold, minLength, lineGap);
	    
	    for (int i = 0; i < lines.total(); i++) {
	    	
	        double[] vec=lines.get(i,0);
	        Point pt1, pt2;
	        
	        pt1=new Point(vec[0],vec[1]);
	        pt2=new Point(vec[2],vec[3]);
	        
	        double gapY = Math.abs(vec[3]-vec[1]);
	        double gapX = Math.abs(vec[2]-vec[0]);
	        
	        if(gapY>limit && limit>0) {
	            Imgproc.line(img, pt1, pt2, new Scalar(0, 0, 0), 10);
	        }
	    }
	}
	
	private static Mat Contour(Mat imageMat) {
		Mat rgb = new Mat();
		rgb = imageMat.clone();
		Mat grayImage = new Mat();
		Mat img_result = new Mat();
		Imgproc.cvtColor(rgb, grayImage, Imgproc.COLOR_RGB2GRAY);
		
		Mat gradThresh = new Mat();
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		
		Imgproc.adaptiveThreshold(grayImage, gradThresh, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 3, 12);
		Imgproc.morphologyEx(gradThresh, gradThresh, Imgproc.MORPH_CLOSE, hierarchy);
		removeVerticalLines(gradThresh, 100);
		Imgproc.findContours(gradThresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
		
		
		if(contours.size() > 0) {
			for(int idx = 0; idx < contours.size(); idx++) {
				Rect rect = Imgproc.boundingRect(contours.get(idx));
				
				// 이미지 크기를 512 x 512 라고 가정
				if(rect.height > 50 && rect.width > 50 && !(rect.width >= 512 - 5 && rect.height >= 512 - 5)) {
					Imgproc.rectangle(imageMat, new Point(rect.br().x - rect.width, rect.br().y - rect.height)
	                        , rect.br()
	                        , new Scalar(0, 255, 0), 5);
					Rect cutImg = new Rect(new Point(rect.br().x - rect.width, rect.br().y - rect.height), rect.br());
					img_result = new Mat(imageMat, cutImg);
				}
			}
			//Imgcodecs.imwrite("doc_original.jpg", rgb);
	        //Imgcodecs.imwrite("doc_gray.jpg", grayImage);
	        //Imgcodecs.imwrite("doc_thresh.jpg", gradThresh);
	        //Imgcodecs.imwrite("doc_contour.jpg", imageMat);
	        Imgcodecs.imwrite("doc_cutimg.jpg", img_result);
		}
		return img_result;
	}
	
	private static void cleanimg(Mat input) {

		Mat blur = new Mat();
		Mat output = new Mat();
		Mat gray = new Mat();
		Mat img_input = input.clone();
		Mat size_up = new Mat();
		
		Imgproc.resize(img_input, size_up, new Size(), 3, 3, Imgproc.INTER_CUBIC);
		Imgproc.cvtColor(size_up, blur, Imgproc.COLOR_RGB2GRAY);
		Imgproc.GaussianBlur(size_up, blur, new Size(3, 3), 0);
		Imgproc.cvtColor(blur, gray, Imgproc.COLOR_RGB2GRAY);
		Imgproc.threshold(gray, output, 200, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
		Imgcodecs.imwrite("sample_result.jpg", output);
		
	}
	
	static Tesseract instance = Tesseract.getInstance();
	
	public static String findText(String fileName) {
		File inputFile = new File(fileName);
		String result = "";
		try {
			result = instance.doOCR(inputFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Mat inputmat = Imgcodecs.imread("input.jpg");
		
		Mat cont = Contour(inputmat);
	
		cleanimg(cont);
		
		System.out.println(findText("sample_result.jpg"));
	}
		
}
