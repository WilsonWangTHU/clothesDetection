PROJECT := rcnntest

CXX := g++
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := test
TOOL_DIR := tool
OBJ_DIR := obj
LIB_DIR := lib
BIN_DIR := bin

CXXFLAGS := -Wall -g -std=c++11
CXXFLAGS += -I$(INCLUDE_DIR) -I../caffe-fast-rcnn/include \
	    -I../caffe-fast-rcnn/build/src \
	    -I/usr/local/cuda-6.5/include
#LDFLAGS := -ldl -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc \
	-lglog -lgflags -lprotobuf -lleveldb -lsnappy -lcblas -latlas \
	-llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lboost_thread -lstdc++ 
LDFLAGS := -ldl ../caffe-fast-rcnn/build/lib/libcaffe.so \
	-lopencv_core -lopencv_highgui -lopencv_imgproc
LDFLAGS += -L$(LIB_DIR) -L../caffe-fast-rcnn/build/lib
LINKFLAGS += -pthread -fPIC 

# includes
INC := $(shell find $(INCLUDE_DIR) -name '*.h' -o -name '*.hpp')

# source
SRC := $(shell find $(SRC_DIR) -name '*.cpp')
TEST_SRC := $(shell find $(TEST_DIR) -name "*.cpp")
TOOL_SRC := $(shell find $(TOOL_DIR) -name "*.cpp")

# objs
OBJ := $(patsubst ${SRC_DIR}/%.cpp, ${OBJ_DIR}/%.o, ${SRC})
TEST_OBJ := $(addsuffix .o, $(addprefix $(OBJ_DIR)/, \
			$(foreach src,$(TEST_SRC),$(basename $(notdir $(src))))))
TOOL_OBJ := $(addsuffix .o, $(addprefix $(OBJ_DIR)/, \
			$(foreach src,$(TOOL_SRC),$(basename $(notdir $(src))))))

# libs
LIB := $(LIB_DIR)/lib$(PROJECT).so
$(info $$LIB is [${LIB}])

# bins
TEST_BIN := $(patsubst ${OBJ_DIR}/%.o, ${BIN_DIR}/%.exe, ${TEST_OBJ})
$(info $$TEST_BIN is [${TEST_BIN}])
TOOL_BIN := $(patsubst ${OBJ_DIR}/%.o, ${BIN_DIR}/%.exe, ${TOOL_OBJ})
$(info $$TOOL_BIN is [${TOOL_BIN}])


all: directories $(LIB)

directories:
	@mkdir -p ${OBJ_DIR}
	@mkdir -p $(dir $(OBJ))
	@mkdir -p $(LIB_DIR)

test: $(TOOL_BIN) $(TEST_BIN)

$(TOOL_BIN): $(BIN_DIR)/%.exe: $(OBJ_DIR)/%.o | $(LIB)
	$(CXX) $< -o $@ $(LINKFLAGS) $(LIB) $(LDFLAGS) 

$(TEST_BIN): $(BIN_DIR)/%.exe: $(OBJ_DIR)/%.o | $(LIB)
	$(CXX) $< -o $@ $(LINKFLAGS) $(LIB) $(LDFLAGS) 

$(TOOL_OBJ): $(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp $(SRC) $(INC)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TEST_OBJ): $(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp $(SRC) $(INC)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(LIB): $(OBJ)
	$(CXX) -shared -o $@ $^ 

$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC)
	$(CXX) $(CXXFLAGS) -fpic -o $@ -c $<

clean:
	rm $(OBJ_DIR)/* $(LIB_DIR)/* $(BIN_DIR)/*
