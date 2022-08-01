#version 310 es

precision highp float;

in vec2 v_position;

const int gridTextureCount = 2;
uniform highp sampler2D u_gridTextures[gridTextureCount];

const int networkHiddenCount = 8;
const int networkOutputCount = 4;
const int networkWeightCount = networkHiddenCount * networkOutputCount;

const int networkHiddenPackedCount = networkHiddenCount / 4;
const int networkWeightPackedCount = networkWeightCount / 4;

uniform vec4 u_networkWeights[networkWeightPackedCount];

out vec4 f_color;

vec4 get_color(vec4 v) {
    const vec4 r = vec4(246.0/255.0, 242.0/255.0, 217.0/255.0, 1.0);
    const vec4 g = vec4(209.0/255.0, 230.0/255.0, 161.0/255.0, 1.0);
    const vec4 b = vec4(164.0/255.0, 216.0/255.0, 235.0/255.0, 1.0);
    if (v.r > v.g) {
        if (v.r > v.b) return r;
        else return b;
    }
    else {
        if (v.g > v.b) return g;
        else return b;
    }
}

void main() {
    vec4 hidden[networkHiddenPackedCount];
    hidden[0] = texture(u_gridTextures[0], v_position);
    hidden[1] = texture(u_gridTextures[1], v_position);

    vec4 outputs = vec4(0.0);
    for (int j = 0; j < networkHiddenPackedCount; j++) {
        vec4 inputValue = tanh(hidden[j]);
        outputs += vec4(
            dot(inputValue, u_networkWeights[j]),
            dot(inputValue, u_networkWeights[networkHiddenPackedCount + j]),
            dot(inputValue, u_networkWeights[2 * networkHiddenPackedCount + j]),
            dot(inputValue, u_networkWeights[3 * networkHiddenPackedCount + j])
        );
    }
    f_color = get_color(outputs);
}
