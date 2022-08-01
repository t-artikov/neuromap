#version 310 es

precision highp float;

in vec2 a_position;
out vec2 v_position;

uniform vec4 u_screenSpaceGridRect;

void main() {
    gl_Position = vec4((a_position * 2.0 - vec2(1.0)) * vec2(1.0, -1.0), 0.0, 1.0);
    v_position = a_position * u_screenSpaceGridRect.zw + u_screenSpaceGridRect.xy;
}
