# This is a helper file to quickly generate some spheres with different bsdf characters and colors

out_file = "disney.xml"

params = ["metallic", "subsurface", "specular", "roughness", "specularTint", "anisotropic", "sheen", "sheenTint", "clearcoat", "clearcoatGloss"]

colors = ["fff100", "ff8c00", "e81123", "ec008c", "68217a", "00188f", "00bcf2", "00b294", "009e49", "bad80a"]
#colors = ["00f000" for x in range(10)]

def hex_val(c):
    if c == "0": return 0
    elif c == "1": return 1
    elif c == "2":return 2
    elif c == "3":return 3
    elif c == "4": return 4
    elif c == "5": return 5
    elif c == "6": return 6
    elif c == "7": return 7
    elif c == "8": return 8
    elif c == "9": return 9
    elif c == "a": return 10
    elif c == "b": return 11
    elif c == "c": return 12
    elif c == "d": return 13
    elif c == "e": return 14
    elif c == "f": return 15
    else:
        print("error")

def color_val(p):
    # color value of p (str, 2)
    if(len(p) == 2):
        p1 = hex_val(p[0])
        p2 = hex_val(p[1])
        return str((p1 * 16 + p2) / 255.0)
    else:
        print("FAIL")

def convert_color_string(c):
    return color_val(c[0:2]) + "," +color_val(c[2:4]) + "," +color_val(c[4:6])

def gen_sphere(x,y, color, param, value):
    return f"""
    <mesh type="sphere">
        <point name="center" value="{x},0,{y}"/>
        <float name="radius" value="0.45"/>
		<bsdf type="disney">
            <color name="baseColor" value="{convert_color_string(color)}"/>
            <float name="{param}" value="{value}"/>
        </bsdf>
    </mesh>
    """

with open(out_file, "w") as wr:
    #write preamble

    wr.write("""<scene>
	<!-- Independent sample generator, user-selected samples per pixel -->
	<sampler type="independent">
		<integer name="sampleCount" value="1024"/>
	</sampler>

	<!-- Use a direct illumination integrator -->
	<integrator type="direct_mats"/>

	<!-- Render the scene as viewed by a perspective camera -->
	<camera type="perspective">
		<transform name="toWorld">
			<lookat target="0,0,0" origin="0,16,0" up="0,0,1"/>
		</transform>

		<!-- Field of view: 40 degrees -->
		<float name="fov" value="40"/>

		<!-- 512 x 512 pixels -->
		<integer name="width" value="512"/>
		<integer name="height" value="512"/>
	</camera>

    <emitter type="envmap">
		<!--<texture type="png_texture" name="albedo">
			<string name="filename" value="../res/canyon1.png"/>
            <boolean name="sphericalTexture" value="true"/>
		</texture>-->
	</emitter>
    """)

    # write all spheres
    for j,y in enumerate(range(-5, -5 + len(params))):
        for i,x in enumerate(range(-5,5)):
            wr.write(gen_sphere(x+0.5 ,y+0.5 , colors[j], params[j], float(i) / 10))


    wr.write("""
</scene>
""")
