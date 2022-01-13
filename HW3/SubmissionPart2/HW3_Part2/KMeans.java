/*** Author :Vibhav Gogate
The University of Texas at Dallas
*****/


import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

import java.util.Random;
import java.awt.*;
 

public class KMeans {
    public static void main(String [] args){
	if (args.length < 3){
	    System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
	    return;
	}
	try{
	    BufferedImage originalImage = ImageIO.read(new File(args[0]));
	    int k=Integer.parseInt(args[1]);
	    BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
	    ImageIO.write(kmeansJpg, "jpg", new File(args[2])); 
	    
	}catch(IOException e){
	    System.out.println(e.getMessage());
	}	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
	int w=originalImage.getWidth();
	int h=originalImage.getHeight();
	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
	Graphics2D g = kmeansImage.createGraphics();
	g.drawImage(originalImage, 0, 0, w,h , null);
	// Read rgb values from the image
	int[] rgb=new int[w*h];
	int count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		rgb[count++]=kmeansImage.getRGB(i,j);
	    }
	}
	// Call kmeans algorithm: update the rgb values
	kmeans(rgb,k);

	// Write the new rgb values to the image
	count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		kmeansImage.setRGB(i,j,rgb[count++]);
	    }
	}
	return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k)
	{
		// initialize k clusters
		Color[] clusters = new Color[k];
		for (int i = 0; i < k; i++)
		{
			int random = new Random().nextInt(rgb.length);
			clusters[i] = new Color(rgb[random]);
		}
		// flag: show if the clusters update
		boolean flag = true;
		
		Color[] rgbColor = new Color[rgb.length];
		for (int i = 0; i < rgb.length; i++)
		{
			rgbColor[i] = new Color(rgb[i]);
		}
				// array to store the index of cluster
		int[] rgbClusterIndex = new int[rgb.length];
		while (flag) 
		{
			flag = false;
			for (int i = 0, size = rgbColor.length; i < size; i++) 
			{
				Color color = rgbColor[i];
				double minDistance = Double.MAX_VALUE;
				int idx = -1;
				for (int j = 0; j < k; j++) 
				{
					Color clusterColor = clusters[j];
					double red = Math.pow(clusterColor.getRed() - color.getRed(), 2);
					double green = Math.pow(clusterColor.getGreen() - color.getGreen(), 2);
					double blue = Math.pow(clusterColor.getBlue() - color.getBlue(), 2);
					double distance = red + green + blue;
					if (distance <= minDistance) 
					{
						minDistance = distance;
						idx = j;
					}
				}
				rgbClusterIndex[i] = idx;
			}
			// update clusters
			for (int i = 0; i < k; i++) 
			{
				Color cluster = clusters[i];
				long red = 0, green = 0, blue = 0;
				int count = 0;
				for (int j = 0; j < rgb.length; j++) 
					if (rgbClusterIndex[j] == i) 
					{
						Color color = rgbColor[j];
						red += color.getRed();
						green += color.getGreen();
						blue += color.getBlue();
						count += 1;
					}
				if (count != 0) 
				{
					int redVar = (int) red / count;
					int greenVar = (int) green / count;
					int blueVar = (int) blue / count;
					if (cluster.getRed() != redVar || cluster.getGreen() != greenVar || cluster.getBlue() != blueVar) 
					{
						flag = true;
						clusters[i] = new Color(redVar, greenVar, blueVar);
					}
				}
			}
		}
		for (int i = 0; i < rgb.length; i++) 
			rgb[i] = clusters[rgbClusterIndex[i]].getRGB();
    }

}
