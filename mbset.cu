/*
 * File:   mbset.cu
 *
 * Created on December 4, 2015
 *
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 *
 * Jonathan Jones
 *
 */

#include <iostream>
#include <string>
#include <stack>
#include <stdio.h>

#include <cuda_runtime_api.h>
#include <GL/freeglut.h>

#include "Complex.cu"


using namespace std;


// Size of window in pixels, both width and height
static const size_t WINDOW_DIM = 512;
// Maximum Iterations
static const size_t MAX_IT = 2000;
// Threshold for detecting a zoom selection
static const unsigned int MIN_AREA_TO_ZOOM = WINDOW_DIM;
// The name of the OpenGL window
static const std::string WINDOW_BASENAME = "Mandelbrot";
// Constants that we use for calculating the buffer areas
static const unsigned int size_C = WINDOW_DIM * WINDOW_DIM * sizeof(Complex);
static const unsigned int size_r = WINDOW_DIM * WINDOW_DIM * sizeof(unsigned int);
// Initial screen coordinates, both host and device
static const Complex minC_i(-2.0, -1.2);
static const Complex maxC_i(1.0, 1.8);


// Forward function declarations
void InitializeColors();
double complex2pixstep_i(const Complex&, const Complex&);
double complex2pixstep_i(const Complex&, const Complex&);
void cord2complex(unsigned int, unsigned int, Complex*);
void pushWindow(const Complex&, const Complex&);
void popWindow();
void DisplayReset();
void Init();
// callback function forward declerations
void DisplayCB();
void MouseCB(int, int, int, int);
void MouseActiveCB(int, int);
void MousePassiveCB(int, int);
void KeyboardCB(unsigned char, int, int);


// Define a class for tracking cursor position/click
// states across the callback functions
class cursorTrack
{
public:
    cursorTrack()
        : x(0), y(0), x_sel(0), y_sel(0), sel_en(false) {}
    int sel_area() {
        return abs(x - x_sel) * abs(y - y_sel);
    }
    GLint x, y, x_sel, y_sel;
    bool sel_en;
};


// Define a class for a view, which consists
// of a cursor, frame, and a pointer to the buffer area.
class view
{
public:
    view() {}
    cursorTrack curs;
    Complex* minC;
    Complex* maxC;
    unsigned int* buf;
    float origin_re;
    float origin_im;
};


// Define the RGB Class
class RGB
{
public:
    RGB()
        : r(0), g(0), b(0), t(255) {}
    RGB(double r0, double g0, double b0, double t0)
        : r(r0), g(g0), b(b0), t(t0) {}
    double r, g, b, t;
};


// a stack of all previous views
std::stack<view> views;

RGB* colors = 0; // Array of color values


// The CUDA kernel that handles all parallel pixel computations
__global__ void mb_pix(Complex* cc, unsigned int* rr)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i = 0;
    Complex Z_init(cc[index]);
    Complex Z_prev(Z_init);
    Complex Z_nxt(Z_init);

    // itterate over the set
    for (; i < MAX_IT; ++i) {
        Z_nxt = (Z_prev * Z_prev) + Z_init;
        if (Z_nxt.magnitude2() > 2.0) break;
        Z_prev = Z_nxt;
    }
    rr[index] = i;
}


// Generate a random array of colors that we will index from the
// number of iterations for each pixel of the mandlebrot
void InitializeColors()
{
    colors = new RGB[MAX_IT + 1];
    for (size_t i = 0; i < MAX_IT; ++i) {
        if (i < 1) {
            colors[i] = RGB(1.0f, 1.0f, 1.0f, 1.0f);
        }
        else {
            double trans = 1.0f;
            double r, g, b;
            if (i < 5) {
                r = 0.0f;
            } else {
                r = 1.0 - drand48();
            }
            if (i < 10) {
                g = 0.0f;
            } else {
                g = 0.6 - static_cast<double>(i) / (4.0f * MAX_IT);
            }
            if (i < 20) {
                b = 1.0;
            } else {
                b = drand48();
            }
            // cout << "r: " << r << ", g: " << g << ", b: " << b << ", t:" << trans << endl;
            colors[i] = RGB(r, g, b, trans);
        }
    }
    colors[MAX_IT] = RGB(); // black
}


// Returns the step value for mapping a pair of complex
// numbers evenly across the window space.
double complex2pixstep_i(const Complex& min, const Complex& max)
{
    return (max.i - min.i) / WINDOW_DIM;
}
double complex2pixstep_r(const Complex& min, const Complex& max)
{
    return (max.r - min.r) / WINDOW_DIM;
}

// convert a set of coordinates into a complex point in the mandlebrot set
void cord2complex(unsigned int x, unsigned int y, Complex* c)
{
    Complex minn(views.top().minC->r, views.top().minC->i);
    Complex maxx(views.top().maxC->r, views.top().maxC->i);
    double re_step = complex2pixstep_r(minn, maxx);
    double im_step = complex2pixstep_i(minn, maxx);
    *c = Complex(views.top().minC->r + (x * re_step), views.top().minC->i + (y * im_step));
}


// callback for keypress - (x,y) coordinate of mouse also given for cb routine
void KeyboardCB(unsigned char key, int x, int y)
{
    switch (key) {
    case 'B':
    case 'b':
        // go back 1 view
        popWindow();
        break;

    default:
        break;
    }
    // repaint the window
    glutPostRedisplay();
}


// callback for mouse click
void MouseCB(int button, int state, int x, int y)
{
    unsigned int area = 0;

    if (state == GLUT_DOWN) {
        switch (button) {
        case GLUT_LEFT_BUTTON:
            views.top().curs.sel_en = true;
            break;

        default:
            break;
        }
    } else if (state == GLUT_UP) {
        switch (button) {
        case GLUT_LEFT_BUTTON:
            views.top().curs.sel_en = false;
            // if the selected area is above our threshold, assume the
            // user meant to select a region to zoom in.
            area = views.top().curs.sel_area();
            if (area > MIN_AREA_TO_ZOOM) {
                // if (1) {
                Complex p1(0, 0);
                Complex p2(0, 0);
                cord2complex(views.top().curs.x, views.top().curs.y, &p1);
                cord2complex(views.top().curs.x_sel, views.top().curs.y_sel, &p2);
                pushWindow(p1, p2);

                cout << "--  p1:\t\t(" << p1.r << "," << p1.i << ")" << endl;
                cout << "--  p2:\t\t(" << p2.r << "," << p2.i << ")" << endl;
                cout << "--  area:\t" << area << " px." << endl;
            }
            break;

        default:
            break;
        }
        // update our current cross-hairs only
        // when a button is released
        views.top().curs.x = x;
        views.top().curs.y = y;
        glutPostRedisplay();
    }
}


// callback when the mouse moves within the window WITHOUT a buttom press
void MousePassiveCB(int x, int y)
{
    // always update our current position if nothing is being selected
    views.top().curs.x = x;
    views.top().curs.y = y;
    // repaint the window
    glutPostRedisplay();
}


// callback when the mouse moves within the window WITH a buttom press
void MouseActiveCB(int x, int y)
{
    double m_x = static_cast<double>(x - views.top().curs.x);
    double m_y = static_cast<double>(y - views.top().curs.y);
    float m = static_cast<float>(m_y / m_x);

    // fixup the y value if the selection is not a square
    if (abs(m) != 1.0) {
        int m_fix = m >= 0 ? 1 : -1;
        int b = static_cast<int>(views.top().curs.y - (m_fix * views.top().curs.x));
        y = m_fix * x + b;
    }

    // set the selection cursor position when we're holding down a button
    views.top().curs.x_sel = x;
    views.top().curs.y_sel = y;
    // repaint the window
    glutPostRedisplay();
}


// callback for display
void DisplayCB()
{
    // init random num gen
    srand48(time(NULL));
    // clear all
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // setup for a new display
    DisplayReset();

    // when we click & drag the cursor, make the overlaid
    // graphics slightly transparent
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // draw in the mandlebrot
    glBegin(GL_POINTS); // single pixel mode
    for (size_t y = 0; y < WINDOW_DIM; ++y) {
        size_t row_id = y * WINDOW_DIM;
        for (size_t x = 0; x < WINDOW_DIM; ++x) {
            unsigned int cii = views.top().buf[row_id + x];
            RGB cc = colors[cii];
            glColor4f(cc.r, cc.g, cc.b, cc.t);
            glVertex2i(x, y);
        }
    }
    glEnd();  // done drawing pixels for mandlebrot

    // draw cross-hair lines over the window
    glLineStipple(1, 0x003f);
    glColor4ub(170, 170, 170, 180);
    glBegin(GL_LINES);
    glVertex2i(0, views.top().curs.y);
    glVertex2i(WINDOW_DIM, views.top().curs.y);
    glEnd();
    glBegin(GL_LINES);
    glVertex2i(views.top().curs.x, 0);
    glVertex2i(views.top().curs.x, WINDOW_DIM);
    glEnd();
    // draw a frame around our current selection if needed
    if (views.top().curs.sel_en == true) {
        glLineStipple(1, 0xffff);
        glColor4ub(255, 0, 0, 60);
        glBegin(GL_POLYGON);
        glVertex2i(views.top().curs.x, views.top().curs.y);
        glVertex2i(views.top().curs.x_sel, views.top().curs.y);
        glVertex2i(views.top().curs.x_sel, views.top().curs.y_sel);
        glVertex2i(views.top().curs.x, views.top().curs.y_sel);
        glEnd();
    }
    glDisable(GL_BLEND);

    glutSwapBuffers();  // for double buffering
}


// sets up the current view for drawing
void DisplayReset()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


// Initialization function wrapper
void Init()
{
    glViewport(0, 0, WINDOW_DIM, WINDOW_DIM);
    glutSetCursor(GLUT_CURSOR_CROSSHAIR); // cause...why not?
    glEnable(GL_LINE_STIPPLE);
    DisplayReset();
    InitializeColors();
}


// Compute the Mandelbrot within the region between
// the given complex numbers
void computeMB(const Complex& min, const Complex& max, unsigned int* res)
{
    // host copy of complex array
    Complex* host_C;
    // device copy of complex array
    Complex* dev_C;
    // device copy of iteration array
    unsigned int* dev_r;
    // Allocate space for device copies of complex array
    cudaMalloc((void**)&dev_C, size_C);
    cudaMalloc((void**)&dev_r, size_r);
    // Allocate memory for the host complex array
    host_C = (Complex*)malloc(size_C);

    // initialize the complex number for each pixel
    for (size_t i = 0; i < WINDOW_DIM; ++i) {
        size_t row_id = WINDOW_DIM * i;
        for (size_t j = 0; j < WINDOW_DIM; ++j) {
            size_t ii = row_id + j;
            host_C[ii] = Complex(min.r + (j * complex2pixstep_r(min, max)), min.i + (i * complex2pixstep_i(min, max)));
        }
    }

    // Copy inputs to device
    cudaMemcpy(dev_C, host_C, size_C, cudaMemcpyHostToDevice);
    // Launch mb_pix() kernel on GPU
    mb_pix <<< WINDOW_DIM, WINDOW_DIM >>> (dev_C, dev_r);
    // Copy result back to host
    cudaMemcpy(res, dev_r, size_r, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(dev_C);
    cudaFree(dev_r);
    free(host_C);
}


// Push a new window computation onto a stack holding all previous
// window views
void pushWindow(const Complex& p1, const Complex& p2)
{
    // construct a new view object
    view v;
    v.buf = (unsigned int*)malloc(size_r);
    // create complex objects with the correct min/max points of each
    v.minC = new Complex(min(p1.r, p2.r), min(p1.i, p2.i));
    v.maxC = new Complex(max(p1.r, p2.r), max(p1.i, p2.i));
    // place it on our stack
    views.push(v);
    // now, compute it
    computeMB(*(v.minC), *(v.maxC), v.buf);
}


// Remove the current window view from our stack, and
// clean up all previsouly allocated memory in the process.
void popWindow()
{
    // only pop something off if we will have at
    // least one frame leftover afterwards.
    if (views.size() > 1) {
        // access our first element
        view v = views.top();
        // cleanup some memory
        delete v.minC;
        delete v.maxC;
        // don't forget the buffer area!
        free(v.buf);
        // and lastly, we pop it off the stack for good
        views.pop();
    }
}


// Main entry point
int main(int argc, char** argv)
{
    // Initialize OPENGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glClearColor(0, 0, 1, 1);

    // get our screen dimensions & set our window name depending on what are parameters are
    const size_t display_width = glutGet(GLUT_SCREEN_WIDTH);
    const size_t display_height = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(WINDOW_BASENAME.c_str());

    // set up the opengl callbacks for display, mouse and keyboard
    Init();
    glutDisplayFunc(DisplayCB);
    glutKeyboardFunc(KeyboardCB);
    glutMouseFunc(MouseCB);
    glutPassiveMotionFunc(MousePassiveCB);
    glutMotionFunc(MouseActiveCB);

    // set our initial view, popping it as the first element of our stack
    pushWindow(minC_i, maxC_i);

    cout << "Displaying Mandelbrot" << endl;
    // This will callback the display, keyboard and mouse
    glutMainLoop();

    return 0;
}
