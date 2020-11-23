#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

/*
 * This file is based on https://www.flipcode.com/archives/HDR_Image_Reader.shtml
 * Changes by Matthias Busenhart
 */

namespace nori
{
	namespace HDRLoader
	{

		typedef unsigned char RGBE[4];

#define MINELEN 8
#define MAXELEN 0x7fff

		struct HDRLoaderResult
		{
			int width, height;
			float* cols;
		};

		float convertComponent(int expo, int val)
		{
			float v = val / 256.0f;
			float d = (float) pow(2, expo);
			return v * d;
		}

		void workOnRGBE(RGBE* scan, int len, float* cols)
		{
			while (len-- > 0)
			{
				int expo = scan[0][3] - 128;
				cols[0] = convertComponent(expo, scan[0][0]);
				cols[1] = convertComponent(expo, scan[0][1]);
				cols[2] = convertComponent(expo, scan[0][2]);
				cols[3] = 0.f; // because PNGTexture wants length 4
				cols += 4;
				scan++;
			}
		}

		bool oldDecrunch(RGBE* scanline, int len, FILE* file)
		{
			int i;
			int rshift = 0;

			while (len > 0)
			{
				scanline[0][0] = static_cast<unsigned char>(fgetc(file));
				scanline[0][1] = static_cast<unsigned char>(fgetc(file));
				scanline[0][2] = static_cast<unsigned char>(fgetc(file));
				scanline[0][3] = static_cast<unsigned char>(fgetc(file));
				if (feof(file))
					return false;

				if (scanline[0][0] == 1 &&
				    scanline[0][1] == 1 &&
				    scanline[0][2] == 1)
				{
					for (i = scanline[0][3] << rshift; i > 0; i--)
					{
						memcpy(&scanline[0][0], &scanline[-1][0], 4);
						scanline++;
						len--;
					}
					rshift += 8;
				}
				else
				{
					scanline++;
					len--;
					rshift = 0;
				}
			}
			return true;
		}

		bool decrunch(RGBE* scanline, int len, FILE* file)
		{
			int i, j;

			if (len < MINELEN || len > MAXELEN)
				return oldDecrunch(scanline, len, file);

			i = fgetc(file);
			if (i != 2)
			{
				fseek(file, -1, SEEK_CUR);
				return oldDecrunch(scanline, len, file);
			}

			scanline[0][1] = static_cast<unsigned char>(fgetc(file));
			scanline[0][2] = static_cast<unsigned char>(fgetc(file));
			i = fgetc(file);

			if (scanline[0][1] != 2 || scanline[0][2] & 128)
			{
				scanline[0][0] = 2;
				scanline[0][3] = static_cast<unsigned char>(i);
				return oldDecrunch(scanline + 1, len - 1, file);
			}

			// read each component
			for (i = 0; i < 4; i++)
			{
				for (j = 0; j < len;)
				{
					unsigned char code = static_cast<unsigned char>(fgetc(file));
					if (code > 128)
					{ // run
						code &= 127;
						unsigned char val = static_cast<unsigned char>(fgetc(file));
						while (code--)
							scanline[j++][i] = val;
					}
					else
					{ // non-run
						while (code--)
							scanline[j++][i] = static_cast<unsigned char>(fgetc(file));
					}
				}
			}

			return feof(file) ? false : true;
		}


		bool load(const char* fileName, HDRLoaderResult& res)
		{
			int  i;
			char str[200];
			FILE* file;

			file = fopen(fileName, "rb");
			if (!file)
				return false;

			fread(str, 10, 1, file);
			if (memcmp(str, "#?RADIANCE", 10))
			{
				fclose(file);
				return false;
			}

			fseek(file, 1, SEEK_CUR);

			char cmd[200];
			i = 0;
			char c = 0, oldc;
			while (true)
			{
				oldc = c;
				c    = static_cast<unsigned char>(fgetc(file));
				if (c == 0xa && oldc == 0xa)
					break;
				cmd[i++] = c;
			}

			char reso[200];
			i = 0;
			while (true)
			{
				c = static_cast<unsigned char>(fgetc(file));
				reso[i++] = c;
				if (c == 0xa)
					break;
			}

			int w, h;
			if (!sscanf(reso, "-Y %d +X %d", &h, &w))
			{
				fclose(file);
				return false;
			}

			res.width  = w;
			res.height = h;

			float* cols = new float[w * h * 4];
			res.cols = cols;

			RGBE* scanline = new RGBE[w];
			if (!scanline)
			{
				fclose(file);
				return false;
			}

			// convert image
			for (int y = h - 1; y >= 0; y--)
			{
				if (decrunch(scanline, w, file) == false)
					break;
				workOnRGBE(scanline, w, cols);
				cols += w * 4;
			}

			delete[] scanline;
			fclose(file);

			return true;
		}

	} // namespace HDRLoader
}