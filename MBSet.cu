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
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors(void)
{
  colors = new RGB[MAX_IT + 1];
  for (size_t i = 0; i < MAX_IT; ++i)
  {
    if (i < 5)
    { // Try this.. just white for small it counts
      colors[i] = RGB(1, 1, 1);
    }
    else
    {
      colors[i] = RGB(drand48(), drand48(), drand48());
    }
  }
  colors[MAX_IT] = RGB(); // black
}

// callback for keypress - (x,y) coordinate of mouse also given for cb routine
void KeyboardCB(unsigned char key, int x, int y)
{
  cout << "Keyboard event:\tkey=" << key << "\tlocation=(" << x << "," << y << ")" << endl;
}

// callback for mouse click
void MouseCB(int button, int state, int x, int y)\
{
  cout << "Mouse event:\tbutton=" << button
  << "\tstate=" << state << "\tlocation=(" << x << "," << y << ")" << endl;
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
}

// callback for display
void DisplayCB(void)
{
  // clear all
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  // Clear the matrix
  glLoadIdentity();
  // Set the viewing transformation
  gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  glutSwapBuffers();  // for double buffering
}


int main(int argc, char** argv)
{
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

  // Set up necessary host and device buffers

  // set up the opengl callbacks for display, mouse and keyboard
  glutDisplayFunc(DisplayCB);
  glutKeyboardFunc(KeyboardCB);
  glutMouseFunc(MouseCB);
  glutPassiveMotionFunc(MousePassiveCB);

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels


  InitializeColors();
  // This will callback the display, keyboard and mouse
  glutMainLoop();

  return 0;
}
