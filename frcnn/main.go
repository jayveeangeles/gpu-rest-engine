package main

// #cgo pkg-config: opencv cudart-10.0
// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lnvinfer_plugin -lglog -lboost_system -lboost_thread -lstdc++fs 
// #cgo CXXFLAGS: -std=c++11 -I.. -O2 -fomit-frame-pointer -Wall
// #include <stdlib.h>
// #include "frcnn.h"
import "C"
import "unsafe"
import "flag"
import "fmt"
import "strconv"

import (
  "io"
  "io/ioutil"
  "log"
  "net/http"
  "time"
  "os"
  "encoding/json"
)

var ctx *C.frcnn_ctx

type ConfigFile struct {
	Model   	string 	`json:"model"`
	Trained   string 	`json:"trained"`
	Label    	string 	`json:"label"`
	TRTModel	string	`json:"trtmodel"`
}

func FRCNNDetect(w http.ResponseWriter, r *http.Request) {
  if r.Method != "POST" {
    http.Error(w, "", http.StatusMethodNotAllowed)
    return
  }

  if err := r.ParseMultipartForm(5 * 1000); err != nil {
    fmt.Fprintf(w, "ParseForm() err: %v", err)
    http.Error(w, err.Error(), http.StatusBadRequest)
    return
  }

  imageFile, header, err := r.FormFile("imagefile")
  if err != nil {
    http.Error(w, err.Error(), http.StatusBadRequest)
    return
  }
  defer imageFile.Close()

  confthre := "0.5"
  if ct := r.FormValue("confthre"); ct != "" {
    confthre = ct
  }

  confThresh, err := strconv.ParseFloat(confthre, 64)

  buffer, err := ioutil.ReadAll(imageFile)
  if err != nil {
    http.Error(w, err.Error(), http.StatusBadRequest)
    return
  }

  start := time.Now()
  cstr, err := C.frcnn_detect(ctx, (*C.char)(unsafe.Pointer(&buffer[0])), C.size_t(len(buffer)), C.float(confThresh))
  if err != nil {
    http.Error(w, err.Error(), http.StatusBadRequest)
    return
  }
  elapsed := time.Since(start)
  log.Printf("Detection took %12s for %#v", elapsed, header.Filename)

  defer C.free(unsafe.Pointer(cstr))

  w.Header().Set("Content-Type", "application/json")
  io.WriteString(w, C.GoString(cstr))
}

func main() {
  var model, trained, label, trtmodel, config string
  var port int

  flag.StringVar(&model,    "model",    "./deploy.pt",        "model prototxt")
  flag.StringVar(&trained,  "trained",  "./model.caffemodel", "trained model")
  flag.StringVar(&label,    "label",    "./cls.txt",          "text file with labels")
  flag.StringVar(&trtmodel, "trtmodel", "./model.trt",        "TensorRT Model")
  flag.StringVar(&config, 	"config", 	"",      							"server config file")

  flag.StringVar(&model,    "m",	"./deploy.pt",        "model prototxt")
  flag.StringVar(&trained,  "t",  "./model.caffemodel", "trained model")
  flag.StringVar(&label,    "l", 	"./cls.txt",          "text file with labels")
  flag.StringVar(&trtmodel, "T", 	"./model.trt",        "TensorRT Model")
  flag.StringVar(&config, 	"c", 	"",      							"server config file")

	flag.IntVar(&port,  "port", 8000, "TensorRT Model")

	flag.IntVar(&port,  "p", 8000, "TensorRT Model")

  flag.Parse()

	if config != "" {
		var configFile ConfigFile

		jsonFile, err := os.Open(config)
		if err != nil {
			log.Fatalln("could not open config file:", err)
			os.Exit(1)
		}

		defer jsonFile.Close()

		// read our opened xmlFile as a byte array.
		byteValue, _ := ioutil.ReadAll(jsonFile)

		// we unmarshal our byteArray which contains our
		// jsonFile's content into 'configFile' which we defined above
		json.Unmarshal(byteValue, &configFile)

		model 		= configFile.Model
		trained		= configFile.Trained
		label 		= configFile.Label
		trtmodel	= configFile.TRTModel
	}

	cmodel    := C.CString(model)
	ctrained  := C.CString(trained)
	clabel    := C.CString(label)
	ctrtmodel := C.CString(trtmodel)

  log.Println("Initializing TensorRT FRCNN Detector")
  var err error
  ctx, err = C.frcnn_initialize(cmodel, ctrained, clabel, ctrtmodel)
  if err != nil {
    log.Fatalln("could not initialize FRCNN Detector:", err)
    return
  }

  defer C.frcnn_destroy(ctx)
  defer C.free(unsafe.Pointer(cmodel))
  defer C.free(unsafe.Pointer(ctrained))
  defer C.free(unsafe.Pointer(clabel))
  defer C.free(unsafe.Pointer(ctrtmodel))

  log.Println("Adding REST endpoint /api/inference")
  http.HandleFunc("/api/inference", FRCNNDetect)
  log.Println(fmt.Sprintf("Starting server listening on :%d", port))
  log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))
}
