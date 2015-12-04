/*
 * File:   MBSet.cu
 *
 * Created on November 22, 2015
 *
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 *
 */

#include <iostream>
#include <string>
#include <sstream>
#include <stack>
#include <stdio.h>

#include <cuda_runtime_api.h>
#include <GL/freeglut.h>

#include "Complex.cu"

using namespace std;

// Size of window in pixels, both width and height
static const size_t WINDOW_DIM = 512;
// Msximum Iterations
static const size_t MAX_IT = 2000;
static const std::string WINDOW_BASENAME = "Mandelbrot";

// Mandlebrot start/end points for c
// static const float x_i = -2.0;
// static const float y_i = -1.2;
// static const float x_f = 1.0;
// static const float y_f = 1.8;

size_t cii = 0;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;

// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

__global__ void mb_pix(Complex* cc, unsigned int* rr)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int i = 0;
  Complex Z_init(cc[index]);
  Complex Z_prev(Z_n);
  Complex Z_nxt;

  // itterate over the set
  for (; i < 2000; ++i) {
    Z_nxt = (Z_prev * Z_prev) + Z_init;

    if (Z_nxt.magnitude2() > 2.0) break;

    Z_prev = Z_nxt;
  }

  rr[index] = i;
}

void InitializeColors(void)
{
  colors = new RGB[MAX_IT + 1];

  for (size_t i = 0; i < MAX_IT; ++i) {
    if (i < 5)
      colors[i] = RGB(1, 1, 1);
    else
      colors[i] = RGB(drand48(), drand48(), drand48());
  }

  colors[MAX_IT] = RGB(); // black
}

// callback for keypress - (x,y) coordinate of mouse also given for cb routine
void KeyboardCB(unsigned char key, int x, int y)
{
  cout << "Keyboard event:\tkey=" << key << "\tlocation=(" << x << "," << y << ")" << endl;
  glutPostRedisplay();
}

// callback for mouse click
void MouseCB(int button, int state, int x, int y)\
{
  cout << "Mouse event:\tbutton=" << button
  << "\tstate=" << state << "\tlocation=(" << x << "," << y << ")" << endl;
  glutPostRedisplay(); // repaint the window

  // Possible buttons - left button is only guaranteed button to exist on a system
  // GLUT_LEFT_BUTTON = 0
  // GLUT_MIDDLE_BUTTON = 1
  // GLUT_RIGHT_BUTTON = 2
  // scroll up = 3
  // scroll down = 4
  // back click = 7
  // forward click = 8

  // Possible states due to release/press
  // GLUT_UP = 1
  // GLUT_DOWN = 0
}

// callback when the mouse moves within the window without a buttom press
void MousePassiveCB(int x, int y)
{
  char buf[12];
  sprintf(buf, "(%u,%u)", x, y);
  std::string newWinName = WINDOW_BASENAME + "\t" + buf;
  glutSetWindowTitle(newWinName.c_str());
  glutPostRedisplay();
}

// callback for display
void DisplayCB(void)
{
  // clear all
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glBegin(GL_POINTS); // single pixel mode

  for (size_t x = 0; x < WINDOW_DIM; ++x) {
    RGB cc = RGB(rand() % 255, rand() % 255, rand() % 255);
    glColor4ub(cc.r, cc.g, cc.b, 255);

    for (size_t y = 0; y < WINDOW_DIM; ++y)
      glVertex2d(x, y);
  }

  glEnd();  // done drawing
  glutSwapBuffers();  // for double buffering
}


void Init(void)
{
  glViewport(0, 0, WINDOW_DIM, WINDOW_DIM);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  InitializeColors();
}


int main(int argc, char** argv)
{
  srand((unsigned int)time(NULL));
  // Initialize OPENGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  // get our screen dimensions & set our window name depending on what are parameters are
  const size_t display_width = glutGet(GLUT_SCREEN_WIDTH);
  const size_t display_height = glutGet(GLUT_SCREEN_HEIGHT);
  glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
  glutInitWindowPosition(100, 100);
  glutCreateWindow(WINDOW_BASENAME.c_str());
  glutSetCursor(GLUT_CURSOR_CROSSHAIR); // cause...why not?

  // set up the opengl callbacks for display, mouse and keyboard
  glutDisplayFunc(DisplayCB);
  glutKeyboardFunc(KeyboardCB);
  glutMouseFunc(MouseCB);
  glutPassiveMotionFunc(MousePassiveCB);
  Init();

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels

  // host copy of complex array
  Complex* host_C;
  // device copy of complex array
  Complex* dev_C;
  // host copy of interation array
  unsigned int* host_r;
  // device copy of iteration array
  unsigned int* dev_r;

  unsigned int size_C = WINDOW_DIM * WINDOW_DIM * sizeof(Complex);
  unsigned int size_r = WINDOW_DIM * WINDOW_DIM * sizeof(unsigned int);

  float realStep = (maxC.r - minC.r) / WINDOW_DIM;
  float imagStep = (maxC.i - minC.i) / WINDOW_DIM;

  // Allocate space for device copies of complex array
  cudaMalloc((void**)&dev_C, size_C);
  cudaMalloc((void**)&dev_r, size_r);

  // Allocate memory for the host complex array
  host_C = (Complex*)malloc(size_C);
  host_r = (unsigned int*)malloc(size_r);

  // initialize the complex number for each pixel
  for (size_t i = 0; i < WINDOW_DIM; ++i) {
    size_t row_id = WINDOW_DIM * i;

    for (size_t j = 0; j < WINDOW_DIM; ++j) {
      size_t ii = row_id + j;

      host_C[ii] = Complex(minC.r + (j * realStep), minC.i + (i * imagStep));
      // cout << "(" << host_C[ii].r << "," << host_C[ii].i << ")  ";
    }

    // cout << endl << endl << endl << endl;
  }

  // Copy inputs to device
  cudaMemcpy(dev_C, &host_C, size_C, cudaMemcpyHostToDevice);
  // Launch mb_pix() kernel on GPU
  mb_pix <<< WINDOW_DIM , WINDOW_DIM>>> (dev_C, dev_r);
  // Copy result back to host
  cudaMemcpy(&host_r, dev_r, size_r, cudaMemcpyDeviceToHost);
  // Cleanup
  cudaFree(dev_C);
  cudaFree(dev_r);
  free(host_C);

  // show iteration array results
  for (size_t i = 0; i < WINDOW_DIM * WINDOW_DIM; ++i) {
    cout << host_r[i] << "\t";
  }

  // InitializeColors();
  // This will callback the display, keyboard and mouse
  glutMainLoop();

  return 0;
}
