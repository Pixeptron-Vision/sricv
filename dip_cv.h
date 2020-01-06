#ifndef  _DIP_CV_H_
#define  __DIP_CV_H_

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
#include "iostream"
using namespace std;
using namespace cv;


int ConditionSet1(int i1, int j1, CvMat* Im);   
int ConditionSet2(int i2, int j2, CvMat* Im);
int Gset1(int i1,int j1,CvMat* Im);
int Gset2(int i2,int j2,CvMat* Im);
CvMat* Nbhd_path(IplImage* I,int i,int j,int N);
void cvMorphThin(IplImage* in, IplImage* ou,int it);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void SegmentGrayImage(IplImage* I, IplImage* Ou,int min, int max,int gray);
int cvNbhd(IplImage* I, int i,int j);
void cvMedAxTransform(IplImage* Input,IplImage* mat,int maskSize);
void Mask(IplImage* P, IplImage* MAT, int i, int j, int ms);
void cvImfill(IplImage* I);
void cvReplaceChannel(IplImage* src,IplImage* dst,int channel);
void cvMedialAxisTransform(IplImage* in, IplImage* mat);
IplImage* Imfill1(IplImage* I);
void variogram(IplImage* I);
IplImage* scRankTrans(IplImage* I, int k);
IplImage* scBorderReplicate(IplImage* I,int k);
IplImage* scNonMaxSup(IplImage* I,int k);
IplImage* scImageTypecast(IplImage* I,int type,int scale);
void scMarkPoints(IplImage*Nms,IplImage* I);
/* Function Description:: 
   1) void CallBackfn   :: Opencv doesn't provide any facility in the image showing window returned by cvShowImage() for viewing the
   Image pixel positions and values. So utilizing mouse call backs, I defined the fn, void CallBackFn to perform a task when a mouse event 
   is encountered on the image showing window
   
   2)IplImage* SegmentGrayImage  :: A user defined fn, to get a binary image(segmented) from the grayscale input image.First argument being input image
     second n third being min and max range of pixels in grayscale image that have to be segmented out. Here throughout my work I considered 8 bit depth images
	 for processing, so binary image has on pixels value=255 and off pixels value=0
	 
   3)cvMorphThin   :: This is a user defined recursive fn, for thinning a binary image.First argument is the input image and second argument is output thinned image
    The third argument, is the number of iterations needed for getting a healthy output thinned image.(Though the fn stops it's recursion at a point where no more furthur processing
	could be done, there may be some loss of detail... So, user has to check the integer of iteration required for o/p he expects)
   4)ConditionSet1 & COnditionSet2 :: These are two supporting functions used by cvMorphThin in performing thinning.They both take the inputs as pixel position co-ordinates and the pointer of Image to be examined.
   In the first iteration, the pixel value being judjed wheather to be deleted or not is tested to satisfy first conditionset. 
   If it satisfies it is removd.The same with the Conditionset2. It is used in 2nd iteration and a pixel is deleted if it satisfies this condition set.Both the functions return integer 1 if the conditions r satisfied, else they return 0  
	 
   5)cvNbhd :: This funtion is used for finding the junctions in the thinned image. It isn't completely implemented yet
   
   */
void scMarkPoints(IplImage* Nms,IplImage* In)
{
	uchar* Nmsdata=(uchar*)Nms->imageData;
	for(int i=0;i<Nms->height;i++)
	{
		for(int j=0;j<Nms->width;j++)
		{
			if (Nmsdata[i*Nms->widthStep+j] > 0)
			{
				//cvRectangle(  In,  cvPoint(i-5,j-5),  cvPoint(i+5,j+5),  cvScalar(0,0,255) );
				  cvCircle (  In,  cvPoint(j,i), 5,  cvScalar(0,0,255), 1, 8 ); 

			}
		}
	}
	cvNamedWindow("Trees Detected");
	cvShowImage("Trees Detected",In);
}

IplImage* scImageTypecast(IplImage* I,int type, int scale)
{
	IplImage* NI=cvCreateImage(cvSize(I->width,I->height),type,I->nChannels);
	uchar* Idata=(uchar*)I->imageData;
	uchar* NIdata=(uchar*)NI->imageData;
	for(int i=0;i<I->height;i++)
	{
		for(int j=0;j<I->width;j++)
		{
			NIdata[i*NI->widthStep+j]=Idata[i*I->widthStep+j]*scale;
		}
	}
	return NI;
}


IplImage* scNonMaxSup(IplImage* I,int k)
{
	int pad=(k-1)/2;
	IplImage* nms=cvCreateImage(cvSize(I->width,I->height),I->depth,I->nChannels);
	cvZero(nms);
	int mi,mj,jmax;
	uchar* data=(uchar*)I->imageData;
	uchar* nms_data=(uchar*)nms->imageData;
	for(int i=pad;i<(I->height);)

	{
		for(int j=pad;j<(I->width);)
		{  
			if(i+pad < I->height && j+pad < I->width)
			{
				 mi=i;mj=j;
				for(int i2=i;i2<i+pad+1;i2++)
				{
					for(int j2=j;j2<j+pad+1;j2++)
					{
						if(data[i2*I->widthStep+j2] > data[i*I->widthStep+j])
						{
							mi=i2;mj=j2;
						}
					}
				}

				for(int i2=i-pad;i2<i+pad+1;i2++)
				{
					
					if(i2 < i)
					{						
						jmax = j+pad+1;
					}
					else
					{
						jmax=j+1;
					}

					for(int j2=j-pad;j2 < jmax;j2++)
					{
						if(data[i2*I->widthStep+j2] > data[mi*I->widthStep+mj])
						{
							break;
							//mi=i2;mj=j2;
						}
					}
				}

				nms_data[mi*nms->widthStep+mj]= data[mi*I->widthStep+mj];
				//nms_data[mi*nms->widthStep+mj]= 255;
			}
		  j=(j+2*pad+1);
		}
	  i=(i+2*pad+1);
	}
	    cvNamedWindow("NMS");
	cvShowImage("NMS",nms);
	return nms;
}




IplImage* scBorderReplicate(IplImage* I,int k)
{
	int pad=(k-1)/2;
	int l;
	IplImage* br=cvCreateImage(cvSize((I->width+2*pad), (I->height+2*pad)), I->depth, 1);
    cvSetImageROI(br, cvRect(pad, pad, (I->width), (I->height)));
	cvCopy(I,br,NULL);
	uchar* Idata=(uchar*)I->imageData;
	uchar* brdata=(uchar*)br->imageData;
	for(int i=0;i<pad;i++)
	{
		for(int j=pad;j<br->width-pad;j++)
		{
			brdata[i*br->widthStep+j]=Idata[j-pad];
		}
	}
	for(int i=(br->height)-pad;i<br->height;i++)
	{
		for(int j=pad;j<br->width-pad;j++)
		{
			brdata[i*br->widthStep+j]=Idata[((br->height)-2*pad)*I->widthStep+j-pad];
		}
	}
	for(int i=0;i<br->height;i++)
	{
		for(int j=0, l=(br->width-1);j<pad;j++,l--)
		{
			brdata[i*br->widthStep+j]=brdata[i*br->widthStep+j+pad];
			brdata[i*br->widthStep+l-j]=brdata[i*br->widthStep+br->width-pad-1];
		}
	}

    cvNamedWindow("BR");
	cvShowImage("BR",br);
	return br;
}


IplImage* scRankTrans(IplImage* I,int k)
{
	int pad,rank;
	pad=(k-1)/2;
    IplImage* br=cvCreateImage(cvSize((I->width+2*pad), (I->height+2*pad)), I->depth, 1);
	cvSet(br,cvScalar(255),NULL);
    cvSetImageROI(br, cvRect(pad, pad, (I->width), (I->height)));
	cvCopy(I,br,NULL);
    IplImage* Rank = cvCreateImage(cvSize(I->width, I->height), IPL_DEPTH_8U, 1);  
	uchar* data=(uchar*)br->imageData;
	uchar* Rank_data=(uchar*)Rank->imageData;	
	for(int i=0;i<(I->height);i++)
	{
		for(int j=0;j<(I->width);j++)
		{
			rank=0;
			 for(int i1=i-pad;i1< i+pad+1;i1++)
	           {
		           for(int j1=j-pad;j1<j+pad+1;j1++)
		            {
						if(data[(i1+pad)*br->widthStep+(j1+pad)] < data[(i+pad)*br->widthStep+(j+pad)])
						{
							rank=rank+1;
							//printf(" %d ",rank);
						}
		            }
	           }
			 Rank_data[(i)*Rank->widthStep+(j)]= rank;
		}
	}
     // cvNormalize(I, I, 0.0, 1.0, NORM_MINMAX);
	 
	return Rank;
}


void variogram(IplImage* I)
{
	assert(I->nChannels==1);

    int row= I->height;
	int col= I->width;
	uchar* Idata =(uchar*)I->imageData;
	int hmax,r,c,N;
	uchar p,pd;
	float res,value;
	//CvMat* var_NESW = cvCreateMat(1,hmax,CV_32FC1);
    //CvMat* var_NWSE = cvCreateMat(1,hmax,CV_32FC1);
	//************************************************* EW direction calculation*******************
	printf("\n ************************************************* EW direction calculation*******************\n");
	if (col > 20)
	{
		 hmax=20;
	}
	else
	{
		 hmax=col-1;
	}
	CvMat* var_EW = cvCreateMat(1,hmax,CV_32FC1);
	//int* var_d = (int*)var->data.ptr;
	for (int delay=1;delay<hmax+1;delay++)
	{
		value=0;
		for( r=0;r<row;r++)
		{
			c=0;
			while((c+delay)< col)
			{					
				p = (uchar)Idata[r*I->widthStep+c];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[(r*(I->widthStep))+(c+delay)];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				c=c++;
			}

		}
		N=(col-delay)*row;
		res=value/(2*N);
		cvmSet(var_EW,0,delay-1,res);
		printf(" %f ",res);

	}
	//************************************ NS direction calculation*************************** 
	printf(" \n************************************************* NS direction calculation*******************\n");
	printf("\n");
	if (row > 20)
	{
		 hmax=20;
	}
	else
	{
		 hmax=row-1;
	}
    CvMat* var_NS = cvCreateMat(1,hmax,CV_32FC1);
	//int* var_d = (int*)var->data.ptr;
	for (int delay=1;delay<hmax+1;delay++)
	{
		value=0;
		for( c=0;c<col;c++)
		{
			r=0;
			while((r+delay)< row)
			{					
				p = (uchar)Idata[r*I->widthStep+c];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[(r*(I->widthStep))+(c+delay)];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				r=r++;
			}

		}
		N=(col-delay)*col;
		res=value/(2*N);
		cvmSet(var_NS,0,delay-1,res);
		printf(" %f ",res);

	}

	//************************************ NE-SW direction calculation*************************** 
	printf("\n ************************************************* NE-SW direction calculation*******************\n");
	int x,y;
	printf("\n");
	if ((col > 20) || (row>20))
	{
		 hmax=20;
	}
	else
	{
		hmax=(row>col ? col-1: row-1);
		 //hmax=row+col-3;
	}
    CvMat* var_NESW = cvCreateMat(1,hmax,CV_32FC1);
	//int* var_d = (int*)var->data.ptr;
	for (int delay=1;delay<hmax+1;delay++)
	{
		value=0;
		for( r=0;r<row;r++)
		{
			
			x=r;y=0;
			while((x-delay >= 0) && (y+delay < col ))
			{					
				p = (uchar)Idata[x*I->widthStep+y];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[((x-delay)*(I->widthStep))+y+delay];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				x=x-delay;y=y+delay;
			}

		}

		for( c=1;c<col;c++)
		{
			
			x=r;y=c;
			while((x-delay >= 0) && (y+delay < col ))
			{					
				p = (uchar)Idata[x*I->widthStep+y];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[((x-delay)*(I->widthStep))+y+delay];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				x=x-delay;y=y+delay;
			}

		}

		N=(col-delay)*col;
		res=value/(2*N);
		cvmSet(var_NESW,0,delay-1,res);
		printf(" %f ",res);

	}


	//************************************ NW-SE direction calculation*************************** 
	printf(" \n************************************************* NW-SE direction calculation*******************\n");
	printf("\n");
	if ((col > 20) || (row>20))
	{
		 hmax=20;
	}
	else
	{
		hmax=(row>col ? col-1: row-1);
		 //hmax=row+col-3;
	}
    CvMat* var_NWSE = cvCreateMat(1,hmax,CV_32FC1);
	//int* var_d = (int*)var->data.ptr;
	for (int delay=1;delay<hmax+1;delay++)
	{
		value=0;
		for( c=col-1;c>=0;c--)
		{
			
			x=0;y=c;
			while((x+delay < row ) && (y+delay <col ))
			{					
				p = (uchar)Idata[x*I->widthStep+y];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[((x+delay)*(I->widthStep))+y+delay];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				x=x+delay;y=y+delay;
			}

		}

		for( r=1;r<row;r++)
		{
			
			x=r;y=c;
			while((x+delay <row ) && (y+delay < col ))
			{					
				p = (uchar)Idata[x*I->widthStep+y];
					//printf(" june, r %d ,d %d ",r,delay);
				pd = (uchar)Idata[((x+delay)*(I->widthStep))+y+delay];
					//printf("july ");
				value=value+abs(p-pd);
				//	printf("Error here %d col, %d c ",col,c);
				x=x+delay;y=y+delay;
			}

		}

		N=(col-delay)*col;
		res=value/(2*N);
		cvmSet(var_NWSE,0,delay-1,res);
		printf(" %f ",res);

	}
	CvMat* var_AVG = cvCreateMat(1,hmax,CV_32FC1);
	printf("\n");
	for (int i1=0;i1<hmax;i1++)
	{
		res=(cvmGet(var_EW,0,i1)+cvmGet(var_NS,0,i1)+cvmGet(var_NESW,0,i1)+cvmGet(var_NWSE,0,i1)/4);
		cvmSet(var_AVG,0,i1,res);
		printf(" %f ",res);
	}


}





void cvReplaceChannel(IplImage* src, IplImage* dst, int channel)
{
	uchar* src_data = (uchar*)src->imageData;
	uchar* dst_data = (uchar*)dst->imageData;
	if (src->depth == dst->depth)
	{
		for (int i = 0; i < src->height; i++)
		{
			for (int j = 0; j < src->width; j++)
			{
				src_data[i*src->widthStep + j + channel] = dst_data[i*dst->widthStep + j];
			}
		}
	}
	else
	{
		printf("Error in cvReplaceChannel..src->depth != dst->depth\n");
	}
}






int cvNbhd(IplImage* I, int i, int j)
{
	int N = 0;
	uchar*ptr1 = (uchar*)I->imageData;
	if (i > 0 && j > 0 && (ptr1[i*I->widthStep + j] == 255))
	{
		int p2 = ((int)ptr1[(i - 1)*I->widthStep + j]) / 255;
		int p3 = ((int)ptr1[(i - 1)*I->widthStep + j + 1]) / 255;
		int p4 = ((int)ptr1[(i)*I->widthStep + j + 1]) / 255;
		int p5 = ((int)ptr1[(i + 1)*I->widthStep + j + 1]) / 255;
		int p6 = ((int)ptr1[(i + 1)*I->widthStep + j]) / 255;
		int p7 = ((int)ptr1[(i + 1)*I->widthStep + j - 1]) / 255;
		int p8 = ((int)ptr1[(i)*I->widthStep + j - 1]) / 255;
		int p9 = ((int)ptr1[(i - 1)*I->widthStep + j - 1]) / 255;
		N = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

	}

	return N;

}


CvMat* Nbhd_path(IplImage* I, int i, int j,int N)
{
	int row;
	row  = 1;
	
	CvMat* mat = cvCreateMat( N+1, 3, CV_32FC1);
	//uchar* mat_d = (uchar*)mat->data.ptr;
	uchar* data = (uchar*)I->imageData;
	cvmSet(mat, 0, 0, N);
	cvmSet(mat,0,1,i);
	cvmSet(mat,0,2,j);
	/*mat_d[0] = N;
	mat_d[1] = i;
	mat_d[2] = j;*/
	int i1, j1;
	for (int i1 = i - 1; i1 < i + 2; i1++)
	{
		for (int j1 = j - 1; j1 < j + 2; j1++)
		{
			
			if (i1 != i || j1 != j)
			{
				
				if (data[i1*I->widthStep + j1] == 255)
				{
					cvmSet(mat, row, 0, row);
					cvmSet(mat, row, 1, i1);
					cvmSet(mat, row, 2, j1);
					/*mat_d[row*mat->step+0] = row;
					mat_d[row*mat->step +1] = i1;
					mat_d[row*mat->step + 2] = j1;*/
					//printf("row no. %d, col no. %d, (i1 %d,j1 %d)\n", mat_d[row*mat->step + 1], mat_d[row*mat->step + 2],i1,j1);
					row = row++;
				}
			}
		}
	}
	return mat;
}


int GSet1(int i1, int j1, CvMat* Im)
{
	int ret, G1,G2,G3,n1,n2;                   //Initializing conditions
	G1 = G2 = G3 = 0;
	uchar* ptr1 = (uchar*)Im->data.ptr;              //Initializing data pointer
	if (i1 > 0 && j1>0)
	{
		// Defining the neighbourhood of the pixel to check for the conditions
		int p3 = ((int)ptr1[(i1 - 1)*Im->step + j1]) / 255;
		int p2 = ((int)ptr1[(i1 - 1)*Im->step + j1 + 1]) / 255;
		int p1 = ((int)ptr1[(i1)*Im->step + j1 + 1]) / 255;
		int p8 = ((int)ptr1[(i1 + 1)*Im->step + j1 + 1]) / 255;
		int p7 = ((int)ptr1[(i1 + 1)*Im->step + j1]) / 255;
		int p6 = ((int)ptr1[(i1 + 1)*Im->step + j1 - 1]) / 255;
		int p5 = ((int)ptr1[(i1)*Im->step + j1 - 1]) / 255;
		int p4 = ((int)ptr1[(i1 - 1)*Im->step + j1 - 1]) / 255;

		// Defining conditions accordingly as per the algorithm
		/*
		G1 = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			(p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
			*/
		int b1 = (~p1) & (p2 | p3);
		int b2 = (~p3) & (p4 | p5);
		int b3 = (~p5) & (p6 | p7);
		int b4 = (~p7) & (p8 | p1);
		G1 = b1 + b2 + b3 + b4;

		//Ap =( p4 || p3 )+(p3||p2)+(p2||p9)+(p9||p8)+(p8||p7)+(p7||p6)+(p6||p5)+(p5||p4) ;
		n1 = (p1 | p2) + (p3 | p4)+(p5 | p6)+(p7 | p8);
		n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
		G2 = (n1 > n2) ? n2 : n1;
		G3 = (p2 | p3 | (~p8)) & p1;

	}

	if (((G2==2) || (G2==3)) && (G1 == 1) && (G3 == 0))   // COnditions to be met for removing the pixel
	{
		ret = 1;
		//printf("Set1 G1 %d,G2 %d,G3 %d\n", G1, G2, G3);
	}
	else
	{
		ret = 0;
	}
	return ret;
	
}

int GSet2(int i2, int j2, CvMat* Im)
{
	int ret,G1,G2,G3,n1,n2;                      // Inintializing conditions
	G1 = G2 = G3 = 0;
	uchar* ptr1 = (uchar*)Im->data.ptr;                 // Initializing data pointer
	if (i2 > 0 && j2>0)
	{
		// Defining the neighbourhood pixels for studying the conditions
		int p3 = ((int)ptr1[(i2 - 1)*Im->step + j2]) / 255;
		int p2 = ((int)ptr1[(i2 - 1)*Im->step + j2 + 1]) / 255;
		int p1 = ((int)ptr1[(i2)*Im->step + j2 + 1]) / 255;
		int p8 = ((int)ptr1[(i2 + 1)*Im->step + j2 + 1]) / 255;
		int p7 = ((int)ptr1[(i2 + 1)*Im->step + j2]) / 255;
		int p6 = ((int)ptr1[(i2 + 1)*Im->step + j2 - 1]) / 255;
		int p5 = ((int)ptr1[(i2)*Im->step + j2 - 1]) / 255;
		int p4 = ((int)ptr1[(i2 - 1)*Im->step + j2 - 1]) / 255;

		// Defining the conditions 
		int b1 = (~p1) & (p2 | p3);
		int b2 = (~p3) & (p4 | p5);
		int b3 = (~p5) & (p6 | p7);
		int b4 = (~p7) & (p8 | p1);		
		G1 = b1 + b2 + b3 + b4;
		/*
		G1 = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
		(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
		(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
		(p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
		*/
		//Ap = (p4 || p3) + (p3 || p2) + (p2 || p9) + (p9 || p8) + (p8 || p7) + (p7 || p6) + (p6 || p5) + (p5 || p4);
		n1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
		n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
		G2 = (n1 > n2) ? n2 : n1;
		G3 = (p6 | p7 | (~p4)) & p5;
	}

	if (((G2==2) || (G2==3)) && (G1 == 1) && (G3 == 0))   // Condition set 2 to be satisfied for removing a pixel in thinning
	{
		ret = 1;		
		//printf("Set2 G1 %d,G2 %d,G3 %d\n", G1, G2, G3);
	}
	else
	{
		ret = 0;
	}

	return ret;

}

void cvImfill(IplImage * I)
{   //Fills the holes (black) in the Image

	IplImage* fill=cvCreateImage(cvSize((I->width+2), (I->height+2)), I->depth, 1);
	cvSet(fill,cvScalar(0),NULL);
    cvSetImageROI(fill, cvRect(1, 1, (I->width), (I->height)));
	cvCopy(I,fill,NULL);
	cvResetImageROI(fill);
	CvConnectedComp comp;
	double A,B;
	double *a,*b; a=&A;b=&B;
	CvPoint P;
	CvPoint* p; p= &P;
	IplImage* ref= cvCloneImage(fill);
	cvMinMaxLoc(ref,a,b,p,NULL,NULL);
	cvFloodFill(fill,P,cvScalar(255),cvScalar(0),cvScalar(250),&comp,( 8 | CV_FLOODFILL_FIXED_RANGE),NULL);
	cvNot(fill,fill);
    cvSetImageROI(fill, cvRect(1, 1, (I->width), (I->height)));
    cvSetImageROI(ref, cvRect(1, 1, (I->width), (I->height)));
	cvOr(fill,ref,I,NULL);
	cvResetImageROI(fill);
	cvResetImageROI(ref);
	//cvFloodFill(I,cvPoint(0,0),cvScalar(255),cvScalar(0),cvScalar(10),&comp,( 8| CV_FLOODFILL_MASK_ONLY | CV_FLOODFILL_FIXED_RANGE | (255<<8) ),NULL);
}








IplImage* cvImfill1(IplImage* I)
{	
	IplImage* U = cvCloneImage(I);
	IplImage* P = cvCloneImage(I);
	//Filling Image using flood fill
	//CvPoint seed = cvPoint(0, 0);
	CvScalar newvalue = cvScalar(255);
	CvScalar lodif = cvScalar(0);
	CvScalar updif = cvScalar(10);
	int flags = 8;
	for (int i = 0; i < I->width; i++)
	{
		uchar* d = (uchar*)U->imageData;
		if (d[i] == 0)
		{
			cvFloodFill(U,cvPoint(i,0),newvalue,lodif,updif,NULL,8,NULL);
		}
		if (d[((I->height) - 1)*I->widthStep + i] == 0)
		{
			cvFloodFill(U, cvPoint(i, I->height-1), newvalue, lodif, updif, NULL, 8, NULL);
		}
	}
	for (int i = 0; i < I->height; i++)
	{
		uchar* d = (uchar*)U->imageData;
		if (d[i*I->widthStep] == 0)
		{
			cvFloodFill(U, cvPoint(0,i), newvalue, lodif, updif, NULL, 8, NULL);
		}
		if (d[I->widthStep*i+I->width-1] == 0)
		{
			cvFloodFill(U, cvPoint( I->width - 1,i), newvalue, lodif, updif, NULL, 8, NULL);
		}
	}
	/*
	cvNamedWindow("Imfilling Image");   
	cvShowImage("Imfilling Image", U);
	waitKey(1);
	*/

	for (int i = 0; i < I->height; i++)
	{
		for (int j = 0; j < I->width; j++)
		{
			uchar* dataFill = (uchar*)P->imageData;
			uchar* data = (uchar*)U->imageData;
			if (data[i*I->widthStep + j] == 0)
			{
				dataFill[i*I->widthStep + j] = 255;
			}
			
		}
	}
	return P;
}








void Mask(IplImage* P,IplImage* MAT, int i, int j,int ms)
{
	int max, val;
	uchar* DtData = (uchar*)P->imageData;
	uchar* Matdata = (uchar*)MAT->imageData;
	max = 0;
	for (int i1 = (i - (ms / 2)); i1 < (i + (ms / 2) + 1); i1++)
	{
		for (int j1 = (j - (ms / 2)); j1 < (j + (ms / 2) + 1); j1++)
		{
			val= (int)DtData[i* P->widthStep + j];									
			if (val>max)
			{
				max = val;
			}			
		}
	}
	for (int i1 = (i - (ms / 2)); i1 < (i + (ms / 2) + 1); i1++)
	{
		for (int j1 = (j - (ms / 2)); j1 < (j + (ms / 2) + 1); j1++)
		{
			Matdata[i1*MAT->widthStep + j1] = 0;
		//	printf("%d val  %d max \n",val,max);
			val = (int)DtData[i* P->widthStep + j];
			if (val==max)
			{
				Matdata[i1*MAT->widthStep + j1] = 255;
			}
			
		}
	}

}

void cvMedialAxisTransform(IplImage* in, IplImage* mat)
{
	float val,run;
	int i,j,N;

	cvZero(mat);
	uchar* mat_data = (uchar*)mat->imageData;
	uchar* in_data = (uchar*)in->imageData;
    IplImage* dt = cvCreateImage(cvSize(in->width, in->height), IPL_DEPTH_32F, 1);
	cvDistTransform(in, dt, CV_DIST_L2, 3, NULL, NULL);
	IplImage* mask = cvCreateImage(cvSize(in->width + 2, in->height + 2), IPL_DEPTH_32F, 1);
	cvSet(mask, cvScalar(0), NULL);
	cvSetImageROI(mask, cvRect(1, 1, in->width, in->height));
	cvCopy(dt, mask, NULL);
	cvResetImageROI(mask);
	uchar* data = (uchar*)mask->imageData;

	cvNormalize(dt, dt, 0.0, 1.0, NORM_MINMAX);
	cvNamedWindow("Dt");
	cvShowImage("Dt", dt);
	for ( i = 1; i < in->height + 1; i++)
	{
		for ( j = 1; j < in->width + 1; j++)
		{
			if (in_data[i*in->widthStep + j] == 255)
			{
				val = (float)data[i*mask->widthStep + j];
				N = 0;
				for (int i1 = i - 1; i1 < i + 2; i1++)
				{
					for (int j1 = j - 1; j1 < j + 2; j1++)
					{
						if (i1 != i || j1 != j)
						{
							run = (float)data[i1*mask->widthStep + j1];
							if (val > run)
							{
								N = N++;
							}
						}
					}
					// printf("      %d N value \n ",N);
				}

				if (N >= 5)
				{
					mat_data[i*mat->widthStep + j] = 255;
				}
			}
		}
		
	}
     
}

void cvMedAxTransform(IplImage* Input, IplImage* mat, int maskSize)
{

	IplImage* Dt = cvCreateImage(cvSize(Input->width, Input->height), IPL_DEPTH_32F, 1);
	IplImage* DtTh = cvCreateImage(cvSize(Input->width, Input->height), IPL_DEPTH_32F, 1);
	cvDistTransform(Input,Dt,CV_DIST_L2,3,NULL,NULL);
	//cvNormalize(Dt, Dt, 0.0, 1.0, NORM_MINMAX);
	for (int i = (maskSize / 2); i <= ((Dt->height) - (maskSize / 2)); (i=i+maskSize))
	{
		for (int j = (maskSize / 2); j <= ((Dt->width) - (maskSize / 2)); (j = j + maskSize))
		{
		//	printf("%d i, %d j\n",i,j);
			Mask(Dt, mat, i, j, maskSize);
		}

	}
	//cvNormalize(Dt, Dt, 0.0, 1.0, NORM_MINMAX);
	cvNamedWindow("Dt");
	cvShowImage("Dt",Dt);
	//void * dt = Dt;
	//cvSetMouseCallback("Distance Transform", CallBackFunc, dt);
	cvMorphThin(Dt,DtTh,30);
	cvNamedWindow("DtTh");
	cvShowImage("DtTh", DtTh);
}

/*
		for (int i1 = i - 1; i1 < i + 2; i1++)
		{
			for (int j1 = j - 1; j1 < j + 2; j1++)
			{
				if (i1 != i || j1 != j)
				{
					if (data[i1*I->widthStep + j1] == 255)
					{
						N = N++;
					}
				}
			}
		}
		*/

void cvMorphThin(IplImage* in, IplImage* ou,int it)
{
	IplImage* Matrix = cvCloneImage(in);
	IplImage* mask = cvCreateImage(cvSize(in->width + 2, in->height + 2), in->depth, 1);
	cvSet(mask,cvScalar(0),NULL);
	cvSetImageROI(mask,cvRect(1,1,in->width,in->height));
	cvCopy(in,mask,NULL);
	cvResetImageROI(mask);

	CvMat Input, *In,*IT,*M;                         // Initializing matrix pointers IT,M

	In = cvGetMat(mask, &Input, 0, 0);                   // COnvert the IplImage* input image into a matrix
	IT = cvCloneMat(In);                               // Define matrix IT,M to be replicas of input image.
	M = cvCloneMat(In);
	cvZero(M);                                         // Make all the elements in matrix M to be zero	
	int c,co = 1;
	/*cvGetImage(IT, mask);                               // Returning the thinned matrix IT data to IplImage ou. It is a sort of CvMat* to IplImage* conversion 
	cvNamedWindow("Many more happy returns of the day swapnil!!");                    // To Watch the process of iteration and it's result output Image
	cvShowImage("Many more happy returns of the day swapnil!!", mask);
	waitKey(10);*/
	do                            // recursive while loop
	{
	//	printf("Before 1st Iteration %d \n", co);
		
		uchar* ptrM = (uchar*)M->data.ptr;             //Define data pointer of the matrices IT,M
		uchar* ptrIT = (uchar*)IT->data.ptr;
		for (int i = 0; i < M->rows; i++)              
		{
			for (int j = 0; j < M->cols; j++)
			{
				if (ptrIT[i*IT->step + j] == 255)
				{
					int k = GSet1(i, j, IT);
					if (k == 1)
					{
						ptrM[i*IT->step + j] = 255;						
					}
				}
			}
		}
		cvSub(IT, M, IT, NULL);                           // Subtraction of matrix M from IT, to remove pixels in IT for thinning and updating IT after the conversion
		c = cvCountNonZero(M);
		/*printf("After 1st Iteration %d , %d  \n", co,c);		
		cvGetImage(IT, mask);                               // Returning the thinned matrix IT data to IplImage ou. It is a sort of CvMat* to IplImage* conversion 
		cvNamedWindow("Many");                    // To Watch the process of iteration and it's result output Image
		cvShowImage("Many", mask);
	    waitKey(1);
		printf("Before 2nd Iteration %d , %d\n", co,c);*/
		//Second Sub iteration
			cvZero(M);                                   // Initializing the count variable and matrix M to zeros
			for (int i = 0; i < M->rows; i++)
			{
				for (int j = 0; j < M->cols; j++)
				{
					if (ptrIT[i*IT->step + j] == 255)
					{
						int k = GSet2(i, j, IT);
						if (k == 1)
						{
							ptrM[i*IT->step + j] = 255;							
						}
					}
				}
			}
			cvSub(IT, M, IT, NULL);                   // Subtracting M from IT and updating IT after removing pixels
			co = co + 1;                              // counting the number of resursive iteration being done
			c = cvCountNonZero(M);
		//	printf("After 2nd Iteration %d , %d\n", co,c);
			cvGetImage(M, mask);
		//	cvNamedWindow("Matrix Image");                    // To Watch the process of iteration and it's result output Image
		//	cvShowImage("Matrix Image", mask);
		//	waitKey(1);

	} while ( (it==0) ? c > 0 : co<it);

	cvGetImage(IT,mask);
	cvSetImageROI(mask,cvRect(1,1,in->width,in->height));
	cvCopy(mask,ou);
	cvResetImageROI(mask);

}


int ConditionSet1(int i1, int j1, CvMat* Im)                 
{
	int ret, Ap, Bp, prod1, prod2;                   //Initializing conditions
	Bp = 0; Ap = 0;
	uchar* ptr1 = (uchar*)Im->data.ptr;              //Initializing data pointer
	if (i1 > 0 && j1>0)
	{
		// Defining the neighbourhood of the pixel to check for the conditions
		int p2 = ((int)ptr1[(i1 - 1)*Im->step + j1])/255;            
		int p3 = ((int)ptr1[(i1 - 1)*Im->step + j1 + 1])/255;
		int p4 = ((int)ptr1[(i1)*Im->step + j1 + 1])/255;
		int p5 = ((int)ptr1[(i1 + 1)*Im->step + j1 + 1])/255;
		int p6 = ((int)ptr1[(i1 + 1)*Im->step + j1])/255;
		int p7 = ((int)ptr1[(i1 + 1)*Im->step + j1 - 1])/255;
		int p8 = ((int)ptr1[(i1)*Im->step + j1 - 1])/255;
		int p9 = ((int)ptr1[(i1 - 1)*Im->step + j1 - 1])/255;

		// Defining conditions accordingly as per the algorithm
		
		Ap = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			
		//Ap =( p4 || p3 )+(p3||p2)+(p2||p9)+(p9||p8)+(p8||p7)+(p7||p6)+(p6||p5)+(p5||p4) ;
		Bp = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;		
		prod1 = p2*p4*p6;
		prod2 = p4*p6*p8;
	}

	if ((2 <= Bp <= 6) && (Ap) == 1 && prod1 == 0 && prod2 == 0)   // COnditions to be met for removing the pixel
	{
		ret = 1;
	}
	else
	{
		ret = 0;	
	}
	return ret;

}

int ConditionSet2(int i2, int j2, CvMat* Im)
{
	int ret, Ap, Bp, prod1, prod2;                      // Inintializing conditions
	Bp = 0; Ap = 0;
	uchar* ptr1 = (uchar*)Im->data.ptr;                 // Initializing data pointer
	if (i2 > 0 && j2>0)
	{
		// Defining the neighbourhood pixels for studying the conditions
		int p2 = ((int)ptr1[(i2 - 1)*Im->step + j2])/255;
		int p3 = ((int)ptr1[(i2 - 1)*Im->step + j2 + 1])/255;
		int p4 = ((int)ptr1[(i2)*Im->step + j2 + 1])/255;
		int p5 = ((int)ptr1[(i2 + 1)*Im->step + j2 + 1])/255;
		int p6 = ((int)ptr1[(i2 + 1)*Im->step + j2])/255;
		int p7 = ((int)ptr1[(i2 + 1)*Im->step + j2 - 1])/255;
		int p8 = ((int)ptr1[(i2)*Im->step + j2 - 1])/255;
		int p9 = ((int)ptr1[(i2 - 1)*Im->step + j2 - 1])/255;

		// Defining the conditions 
		
		Ap = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			
		//Ap = (p4 || p3) + (p3 || p2) + (p2 || p9) + (p9 || p8) + (p8 || p7) + (p7 || p6) + (p6 || p5) + (p5 || p4);
		Bp = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;		
		prod1 = p2*p4*p8;
		prod2 = p2*p6*p8;
	}

	if ((2 <= Bp <= 6) && Ap == 1 && prod1 == 0 && prod2 == 0)   // Condition set 2 to be satisfied for removing a pixel in thinning
	{
		ret = 1;		
	}
	else
	{
		ret = 0;
	}
	return ret;

}

// Function for using mouse events to display pixel position & intensity
// Only EVENT_MOUSEMOVE is utilized to display the pixel position & it's intensity
void CallBackFunc(int event, int x, int y, int flags, void*userdata)  
{
	if (event == EVENT_LBUTTONDOWN)
	{
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		IplImage* g = (IplImage*)userdata;     // Get the Image from the argument userdata
		uchar* Im = (uchar*)g->imageData;      // Define data pointer of the Image
		switch (g->nChannels)
		{
		case 1:
			{
		       float Intensity = (float)Im[y*g->widthStep + x*g->nChannels + 0];  // Accessing intensity of the pixel position, where mouse pointer is placed			
		       //cout << "Pixel position (" << x << ", " << y << ")-Intensity(" << Intensity << ")" << endl;		// Display the pixel position & Intensity
		       printf("Pixel Position-(x,y):: ( %d , %d ),Intensity- %f \n",x,y,Intensity);
			}
		case 3:
			{
				float IntensityB = (float)Im[y*g->widthStep + x*g->nChannels + 0];
				float IntensityG = (float)Im[y*g->widthStep + x*g->nChannels + 1];
				float IntensityR = (float)Im[y*g->widthStep + x*g->nChannels + 2];
				printf("Pixel Position-(x,y):: ( %d , %d ),Intensity- (B/H,G/S,R/V) =  (%.2f ,%.2f ,%.2f )  \n",x,y,IntensityB,IntensityG,IntensityR);
			}

		}
	}
}


void SegmentGrayImage(IplImage* I, IplImage* Ou,int min, int max,int gray)
{
	//Segments the image as per given range of threshold
	//If gray is 0 the segmented portion is given 255 else it retains original value 
	assert (I->nChannels == 1);                                                               // Check if the input image is gray scale image or not	
	IplImage* bw = cvCreateImage(cvSize(I->width, I->height), I->depth, I->nChannels);   // Define Image pointer for the input image
				uchar* p = (uchar*)I->imageData;	
				uchar* q = (uchar*)bw->imageData;	
  if(Ou==NULL)
  {
	  
		for (int i = 0; i < I->height; i++)
		{
			for (int j = 0; j < I->width; j++)
			{
	
				int pix = (int)p[i*I->widthStep + j];
			   
				 if (pix >= min && pix <= max)
				   {
					  if(gray==0)
					  {
					    q[i*bw->widthStep + j] = 255;			
					  }
					 else
					  {
						 q[i*bw->widthStep + j]=pix;
					  }
				   }
				   else
				  {
					q[i*bw->widthStep + j] = 0;					
				  }			   
			}  			
		}
		Ou=bw;
		//cvCopy(bw,Ou,NULL);
   }
else
	{
		//printf("Yes I am Here , %d  , %d \n",min,max);
		uchar* pa = (uchar*)Ou->imageData;	
		for (int i = 0; i < I->height; i++)
		{
			for (int j = 0; j < I->width; j++)
			{
	
				int pix = (int)p[i*I->widthStep + j];
			   
				 if (pix >= min && pix <= max)
				   {
					  if(gray==0)
					  {
					    pa[i*bw->widthStep + j] = 255;			
					  }
					  else
					  {
						  pa[i*bw->widthStep + j] =(int)p[i*I->widthStep + j] ;
					  }
					 
				   }
				   else
				  {
					  pa[i*bw->widthStep + j] = 0;					
				  }			   
			}  			
		}
	}
}



#endif  
















/*
if (g->origin == IPL_ORIGIN_BL)
{
cout << "Mouse - position (" << g->width-x << ", " << g->height-y << ")" << endl;
}
else
{
cout << "Mouse - position (" << x << ", " << y << ")" << endl;
}
//int d = g->widthStep;
//cout << "Pixel position (" << x << ", " << y << ")-Widthstep(" << d << ")" << endl;
*/