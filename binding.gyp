{
	"targets": [
		{
			"target_name": "addon-floatfann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc",
				"src/training-data.cc"
			],
			"libraries": [
				"../fann/lib/libfloatfann.a"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")", "fann/include" ],
			"defines": [
				"FANNY_FLOAT"
			],
			"cflags_cc": [ "-fPIC" ]
		},
		{
			"target_name": "addon-doublefann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc",
				"src/training-data.cc"
			],
			"libraries": [
				"../fann/lib/libdoublefann.a"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")", "fann/include" ],
			"defines": [
				"FANNY_DOUBLE"
			],
			"cflags_cc": [ "-fPIC" ]
		},
		{
			"target_name": "addon-fixedfann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc",
				"src/training-data.cc"
			],
			"libraries": [
				"../fann/lib/libfixedfann.a"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")", "fann/include" ],
			"defines": [
				"FANNY_FIXED"
			],
			"cflags_cc": [ "-fPIC" ]
		}

	]
}
