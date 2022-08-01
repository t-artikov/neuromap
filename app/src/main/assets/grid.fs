#version 310 es
#extension GL_EXT_texture_buffer : require

precision highp float;
precision highp int;

in vec2 v_position;

uniform highp samplerBuffer u_encodingTexture;

uniform int u_level;
uniform float u_levelOpacity;

const int encodingLevelCount = 12;
const int encodingFeaturesPerLevel = 4;
const int encodingCount = encodingLevelCount * encodingFeaturesPerLevel;
const int encodingPackedCount = encodingCount / 4;
uniform int u_encodingResolutions[encodingLevelCount];
uniform int u_encodingParamCounts[encodingLevelCount];
uniform int u_encodingParamOffsets[encodingLevelCount];

const int networkInputCount = encodingCount;
const int networkInputPackedCount = networkInputCount / 4;
const int networkHiddenCount = 8;
const int networkHiddenPackedCount = networkHiddenCount / 4;
const int networkWeightCount = networkInputCount * networkHiddenCount;
const int networkWeightPackedCount = networkWeightCount / 4;
uniform vec4 u_networkWeights[networkWeightPackedCount];

layout(location = 0) out vec4 f_color0;
layout(location = 1) out vec4 f_color1;


uint hash(uint x, uint y, uint resolution, uint paramCount) {
    if (resolution * resolution > paramCount) {
        return (y * 2654435761u) ^ x;
    } else {
        return y * resolution + x;
    }
}

vec4 getEncoding(int x, int y, int level) {
    uint resolution = uint(u_encodingResolutions[level]);
    uint paramCount = uint(u_encodingParamCounts[level]);
    uint paramOffset = uint(u_encodingParamOffsets[level]);

    uint index = hash(uint(x), uint(y), resolution, paramCount) % paramCount;
    return texelFetch(u_encodingTexture, int(index + paramOffset));
}

vec4 getEncodingInterpolated(vec2 position, int level) {
    if (level > u_level) {
        return vec4(0.0);
    }
    int resolution = u_encodingResolutions[level];
    vec2 scaledPosition = position * float(resolution - 1);
    vec2 intPosition;
    vec2 fracPosition = modf(scaledPosition, intPosition);
    int ix = int(intPosition.x);
    int iy = int(intPosition.y);

    vec4 v00 = getEncoding(ix, iy, level);
    vec4 v10 = getEncoding(ix + 1, iy, level);
    vec4 v01 = getEncoding(ix, iy + 1, level);
    vec4 v11 = getEncoding(ix + 1, iy + 1, level);

    vec4 v0 = mix(v00, v10, fracPosition.x);
    vec4 v1 = mix(v01, v11, fracPosition.x);
    return mix(v0, v1, fracPosition.y);
}

void main() {
    vec2 position = clamp(v_position, vec2(0.0), vec2(1.0));

    vec4 encodings[encodingPackedCount];
    for (int i = 0; i < encodingLevelCount; i++) {
        encodings[i] = getEncodingInterpolated(position, i);
    }
    encodings[u_level] *= u_levelOpacity;

    vec4 hidden[networkHiddenPackedCount];
    for (int i = 0; i < networkHiddenPackedCount; i++) {
        hidden[i] = vec4(0.0);
        for (int j = 0; j < networkInputPackedCount; j++) {
            vec4 inputValue = encodings[j];
            hidden[i] += vec4(
                dot(inputValue, u_networkWeights[(i * 4) * networkInputPackedCount + j]),
                dot(inputValue, u_networkWeights[(i * 4 + 1) * networkInputPackedCount + j]),
                dot(inputValue, u_networkWeights[(i * 4 + 2) * networkInputPackedCount + j]),
                dot(inputValue, u_networkWeights[(i * 4 + 3) * networkInputPackedCount + j])
            );
        }
    }
    f_color0 = hidden[0];
    f_color1 = hidden[1];
}
