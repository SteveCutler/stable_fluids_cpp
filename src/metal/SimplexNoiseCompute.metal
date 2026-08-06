#include <metal_stdlib>
using namespace metal;

/// Permutation table. Random jumble of all numbers 0-255.
/// Produces a repeatable pattern of 256, but this is not a problem
/// for graphic textures as the noise features disappear at a distance
/// far enough to be able to see a repeatable pattern of 256.
constant uint8_t perm[256] = {
    87, 34, 171, 119, 15, 162, 234, 180, 116, 5, 161, 43, 126, 163, 117, 193,
    96, 218, 86, 47, 29, 84, 189, 199, 64, 2, 197, 10, 120, 195, 127, 59,
    103, 198, 98, 23, 207, 51, 200, 33, 155, 148, 107, 187, 236, 131, 216, 239,
    156, 182, 48, 113, 210, 229, 128, 212, 149, 204, 153, 111, 58, 147, 71, 118,
    36, 158, 179, 241, 74, 62, 122, 41, 83, 253, 244, 228, 9, 85, 16, 112,
    141, 135, 186, 53, 165, 157, 27, 188, 20, 101, 72, 4, 175, 17, 211, 70,
    136, 32, 173, 1, 75, 151, 26, 81, 150, 223, 183, 252, 93, 95, 65, 206,
    67, 18, 68, 90, 66, 185, 245, 37, 12, 191, 91, 137, 242, 176, 205, 255,
    190, 222, 220, 77, 250, 196, 177, 134, 192, 168, 133, 19, 94, 45, 219, 164,
    60, 170, 208, 221, 217, 154, 194, 159, 169, 55, 209, 8, 61, 76, 146, 22,
    240, 246, 184, 152, 167, 144, 3, 106, 99, 109, 50, 233, 115, 238, 110, 44,
    174, 108, 249, 31, 54, 97, 121, 69, 202, 56, 105, 251, 237, 57, 224, 226,
    203, 247, 104, 30, 140, 7, 132, 11, 49, 40, 24, 129, 139, 227, 166, 143,
    231, 82, 14, 46, 28, 39, 13, 172, 124, 138, 73, 178, 160, 0, 142, 88,
    213, 230, 130, 78, 243, 125, 145, 181, 215, 232, 201, 89, 92, 102, 25, 235,
    248, 42, 79, 254, 114, 123, 214, 35, 38, 63, 225, 21, 52, 100, 6, 80
};

/// Gradient lookup table for branchless gradient computation.
/// 12 unique gradient directions, with indices 12-15 repeating earlier entries.
/// Using half3 for better performance on Apple GPUs.
constant half3 grad3[16] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 0}, {-1, 1, 0}, { 0,-1, 1}, { 0,-1,-1}
};



/// Simplex corner offset lookup tables.
/// Indexed by comparison results: (x0>=y0)*4 + (y0>=z0)*2 + (x0>=z0)
/// Replaces nested if/else branching with constant memory lookup.
///
/// Note: indices 1 and 6 are geometrically unreachable:
///   - Index 1 (x<y, y<z, x>=z): x<y<z implies x<z, contradicts x>=z
///   - Index 6 (x>=y, y>=z, x<z): x>=y>=z implies x>=z, contradicts x<z
constant int3 simplex1[8] = {
    {0, 0, 1},  // 0: x<y, y<z, x<z   -> Z Y X
    {0, 0, 1},  // 1: unreachable
    {0, 1, 0},  // 2: x<y, y>=z, x<z  -> Y Z X
    {0, 1, 0},  // 3: x<y, y>=z, x>=z -> Y X Z
    {0, 0, 1},  // 4: x>=y, y<z, x<z  -> Z X Y
    {1, 0, 0},  // 5: x>=y, y<z, x>=z -> X Z Y
    {1, 0, 0},  // 6: unreachable
    {1, 0, 0}   // 7: x>=y, y>=z, x>=z -> X Y Z
};

constant int3 simplex2[8] = {
    {0, 1, 1},  // 0: Z Y X
    {0, 1, 1},  // 1: unreachable
    {0, 1, 1},  // 2: Y Z X
    {1, 1, 0},  // 3: Y X Z
    {1, 0, 1},  // 4: Z X Y
    {1, 0, 1},  // 5: X Z Y
    {1, 0, 1},  // 6: unreachable
    {1, 1, 0}   // 7: X Y Z
};


// MARK: - Helper Functions

/// Computes the largest integer value not greater than the float one.
/// This method is faster than using (int32_t)std::floor(fp).

inline int fastfloor(float fp)
{
    int i = int(fp);
    return (fp < float(i)) ? i - 1 : i;
}

inline uchar hash(int i)
{
    return perm[uchar(i)];
}

inline half grad(int h, half x, half y, half z)
{
    half3 g = grad3[h & 15];
    return fma(g.x, x, fma(g.y, y, g.z * z));
}


// MARK: - Noise Functions

/// 3D Perlin simplex noise.
/// Optimized for mobile GPUs with branchless operations and half precision.
///
/// @param x float coordinate
/// @param y float coordinate
/// @param z float coordinate
/// @return Noise value in the range [-1, 1], value of 0 on all integer coordinates.
float noise3D(float x, float y, float z) {
    // Skewing/Unskewing factors for 3D
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;
    const float G3_2 = 2.0f / 6.0f;
    const float G3_3 = 3.0f / 6.0f;

    // Skew the input space to determine which simplex cell we're in
    float s = (x + y + z) * F3;
    int i = fastfloor(x + s);
    int j = fastfloor(y + s);
    int k = fastfloor(z + s);
    float t = (i + j + k) * G3;

    // Unskew the cell origin back to (x,y,z) space
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;

    // The x,y,z distances from the cell origin
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    // Determine simplex corner offsets using lookup table (branchless)
    int cmp = (x0 >= y0 ? 4 : 0) + (y0 >= z0 ? 2 : 0) + (x0 >= z0 ? 1 : 0);
    int3 off1 = simplex1[cmp];
    int3 off2 = simplex2[cmp];

    // Offsets for corners in (x,y,z) coords
    float x1 = x0 - off1.x + G3;
    float y1 = y0 - off1.y + G3;
    float z1 = z0 - off1.z + G3;
    float x2 = x0 - off2.x + G3_2;
    float y2 = y0 - off2.y + G3_2;
    float z2 = z0 - off2.z + G3_2;
    float x3 = x0 - 1.0f + G3_3;
    float y3 = y0 - 1.0f + G3_3;
    float z3 = z0 - 1.0f + G3_3;

    // Hashed gradient indices of the four simplex corners
    int gi0 = hash(i + hash(j + hash(k)));
    int gi1 = hash(i + off1.x + hash(j + off1.y + hash(k + off1.z)));
    int gi2 = hash(i + off2.x + hash(j + off2.y + hash(k + off2.z)));
    int gi3 = hash(i + 1 + hash(j + 1 + hash(k + 1)));

    // Calculate contributions from four corners (branchless using max)
    half t0 = max((half)0.0h, (half)(0.6f - x0*x0 - y0*y0 - z0*z0));
    t0 *= t0;
    half n0 = t0 * t0 * grad(gi0, (half)x0, (half)y0, (half)z0);

    half t1 = max((half)0.0h, (half)(0.6f - x1*x1 - y1*y1 - z1*z1));
    t1 *= t1;
    half n1 = t1 * t1 * grad(gi1, (half)x1, (half)y1, (half)z1);

    half t2 = max((half)0.0h, (half)(0.6f - x2*x2 - y2*y2 - z2*z2));
    t2 *= t2;
    half n2 = t2 * t2 * grad(gi2, (half)x2, (half)y2, (half)z2);

    half t3 = max((half)0.0h, (half)(0.6f - x3*x3 - y3*y3 - z3*z3));
    t3 *= t3;
    half n3 = t3 * t3 * grad(gi3, (half)x3, (half)y3, (half)z3);

    // Sum contributions and scale to [-1,1]
    return 32.0f * (float)(n0 + n1 + n2 + n3);
}



/// Fractal/Fractional Brownian Motion (fBm) summation of 3D Perlin Simplex noise.
///
/// @param octaves  Number of fractions of noise to sum
/// @param x        float coordinate
/// @param y        float coordinate
/// @param z        float coordinate
/// @param freq     Noise frequency
/// @param amp      Noise amplitude
/// @param lac      Lacunarity (frequency multiplier per octave)
/// @param per      Persistence (amplitude multiplier per octave)
/// @return Noise value in the range [-1, 1]
float fractal3D(uint octaves, float x, float y, float z, float freq, float amp, float lac, float per) {
    float output = 0.0f;
    float denom = 0.0f;

    for (uint i = 0; i < octaves; i++) {
        output = fma(amp, noise3D(x * freq, y * freq, z * freq), output);
        denom += amp;
        freq *= lac;
        amp *= per;
    }

    return output / denom;
}




kernel void generateNoise(
    device float* noise       [[buffer(0)]],
    constant float& time      [[buffer(1)]],
    constant float& frequency [[buffer(2)]],
    constant uint& width      [[buffer(3)]],
    constant uint& height     [[buffer(4)]],
    uint2 gid                 [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint index = gid.y * width + gid.x;

    float x = float(gid.x) * frequency;
    float y = float(gid.y) * frequency;
    float z = time * frequency;

    // Already approximately in [-1, 1]
    noise[index] = noise3D(x, y, z);
}
