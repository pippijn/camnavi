using System;
using System.IO;
using System.Drawing;
using System.Runtime.InteropServices;

namespace rec.robotino.com
{
/**
 * 
 */
sealed class ImageLoader
{
    public static Image loadImage(IntPtr dataPtr, uint dataSize, uint width, uint height, uint numChannels, uint bitsPerChannel, uint step)
    {
        byte[] data = new byte[dataSize];
        Marshal.Copy(dataPtr, data, 0, (int)dataSize);
        
        return loadRGBImage(data, (int)width, (int)height, (int)step);
    }

    private static Image loadRGBImage(byte[] rawData, int width, int height, int step )
    {
        Bitmap bitmap = new Bitmap(width, height);

        for (int line = 0; line < height; line++)
        {
            for (int x = 0; x < width; x++)
            {
                byte r = rawData[(x*3 + line * step) + 0];
                byte g = rawData[(x*3 + line * step) + 1];
                byte b = rawData[(x*3 + line * step) + 2];

                bitmap.SetPixel(x, line, Color.FromArgb(r, g, b));
            }
        }

        return bitmap;
    }
}
}
