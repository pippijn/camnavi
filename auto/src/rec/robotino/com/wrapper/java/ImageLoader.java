package rec.robotino.com;

import java.awt.Point;
import java.awt.Image;
import java.awt.image.*;
import java.awt.color.ColorSpace;
import java.io.ByteArrayInputStream;
import javax.imageio.ImageIO;

/**
 * 
 */
final class ImageLoader
{
    private static BufferedImage image;

    public static Image loadImage( byte[] data, long width, long height, long numChannels, long bitsPerChannel, long step)
    {
        return loadRGBImage(data, (int)width, (int)height, (int)step);
    }

    private static Image loadRGBImage(byte[] rawData, int width, int height, int step)
    {
        if (image == null || image.getWidth() != width || image.getHeight() != height)
        {
            WritableRaster raster = Raster.createWritableRaster(new BandedSampleModel(DataBuffer.TYPE_BYTE, width, height, 3), new Point(0, 0));
            ColorModel colorModel = new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_sRGB), false, false, ColorModel.OPAQUE, DataBuffer.TYPE_BYTE);
            image = new BufferedImage(colorModel, raster, false, null);
        }

        byte[][] buffer = ((DataBufferByte) image.getRaster().getDataBuffer()).getBankData();

        for (int i = 0; i < buffer[0].length; i++)
        {
            buffer[0][i] = rawData[i * 3];
            buffer[1][i] = rawData[i * 3 + 1];
            buffer[2][i] = rawData[i * 3 + 2];
        }
        
        /**
        for (int line = 0; line < height; line++)
        {
            for (int x = 0; x < width; x++)
            {
                buffer[0][x + line * width ] = rawData[(x*3 + line * step)];
				buffer[1][x + line * width ] = rawData[(x*3 + line * step) + 1];
				buffer[2][x + line * width ] = rawData[(x*3 + line * step) + 2];
            }
        }*/
        
        return image;
    }
} 

