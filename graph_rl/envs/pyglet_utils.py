import numpy as np
import pyglet.gl as gl

from .graphics_utils import ArrowConfig

def draw_circle_sector(center, angle, radius, n, color, triangles_to_draw):
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glColor3f(*color)
    gl.glVertex2f(*center)
    for i in range(triangles_to_draw + 1):
        gl.glVertex3f(center[0] + np.cos(2.*np.pi/n*i + angle)*radius, 
                      center[1] + np.sin(2.*np.pi/n*i + angle)*radius,
                      0.0)
    gl.glEnd()

def draw_box(center, diameter, length, phi, color):
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(center[0], center[1], 0.)
    gl.glRotatef(phi, 0., 0., 1.)
    gl.glBegin(gl.GL_QUADS)
    gl.glColor3f(*color)
    gl.glVertex2f(-0.5*diameter, -0.5*length)
    gl.glVertex2f(0.5*diameter, -0.5*length)
    gl.glVertex2f(0.5*diameter, 0.5*length)
    gl.glVertex2f(-0.5*diameter, 0.5*length)
    gl.glEnd()
    gl.glPopMatrix()

def draw_line(center, radius, angle, line_color):
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(*line_color)
    gl.glVertex2f(center[0] - np.cos(angle)*radius, 
    center[1] - np.sin(angle)*radius)
    gl.glVertex2f(center[0] - np.cos(angle + np.pi)*radius, 
    center[1] - np.sin(angle + np.pi)*radius)
    gl.glEnd()


def draw_vector(initial_point, vector, arrow_config):
    v = vector*arrow_config.scale
    length = np.linalg.norm(v)
    width = arrow_config.width
    arrow_head_size = arrow_config.head_size
    color = arrow_config.color
    # orthogonal vector used for constructing vertices that make up 
    # arrow shape
    w = np.array([v[1]/length, -v[0]/length])
    # factor scaling vector such that it only reaches to arrow head and not to tip
    rescale = (length - arrow_config.head_size)/length
    v_re = np.array(v)*rescale

    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(initial_point[0], initial_point[1], 0.)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor3f(*color)
    # rectangle
    gl.glVertex2f( 0.5*w[0]*width,              0.5*w[1]*width)
    gl.glVertex2f( 0.5*w[0]*width + v_re[0],  0.5*w[1]*width + v_re[1])
    gl.glVertex2f(-0.5*w[0]*width + v_re[0], -0.5*w[1]*width + v_re[1])
    gl.glVertex2f(-0.5*w[0]*width + v_re[0], -0.5*w[1]*width + v_re[1])
    gl.glVertex2f(-0.5*w[0]*width,             -0.5*w[1]*width)
    gl.glVertex2f( 0.5*w[0]*width,              0.5*w[1]*width)
    # arrow head
    gl.glVertex2f( 0.5*w[0]*arrow_head_size + v_re[0],  
            0.5*w[1]*arrow_head_size + v_re[1])
    gl.glVertex2f(-0.5*w[0]*arrow_head_size + v_re[0], 
            -0.5*w[1]*arrow_head_size + v_re[1])
    gl.glVertex2f(v[0], v[1])
    gl.glEnd()
    gl.glPopMatrix()

def draw_vector_with_outline(initial_point, vector, arrow_config, outline_color, inner_shrink_factor = 0.6):
    ac_outline = ArrowConfig(arrow_config.scale, arrow_config.width, arrow_config.head_size, outline_color)
    length = np.linalg.norm(vector) 
    scale_inner = (arrow_config.scale*length - arrow_config.head_size*(1. - inner_shrink_factor))/length
    ac_inner = ArrowConfig(scale_inner, arrow_config.width*inner_shrink_factor, 
        arrow_config.head_size*inner_shrink_factor, arrow_config.color)
    draw_vector(initial_point, vector, ac_outline)
    draw_vector(initial_point, vector, ac_inner)

def draw_circular_subgoal(position, velocity, radius, color, arrow_config, n_triangles = 32):
    draw_circle_sector(position, 
            0., 
            radius,
            n_triangles, 
            (0., 0., 0.),
            n_triangles)
    draw_circle_sector(position, 
            0., 
            0.8*radius,
            n_triangles, 
            color,
            n_triangles)
    if velocity is not None:
        draw_vector_with_outline(position, velocity, arrow_config, (0., 0., 0.))
