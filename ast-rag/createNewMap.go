package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Field represents a single parameter, return value, or struct field.
type Field struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Tag  string `json:"tag,omitempty"`
}

// FunctionMetadata holds information for a single function.
type FunctionMetadata struct {
	Name       string  `json:"name"`
	Signature  string  `json:"signature"`
	Parameters []Field `json:"parameters"`
	Returns    []Field `json:"returns"`
	Body       string  `json:"body"`
}

// StructMetadata holds information for a single struct.
type StructMetadata struct {
	Name   string  `json:"name"`
	Fields []Field `json:"fields"`
}

// NEW: Method represents a single method within an interface.
type Method struct {
	Name       string  `json:"name"`
	Signature  string  `json:"signature"`
	Parameters []Field `json:"parameters"`
	Returns    []Field `json:"returns"`
}

// NEW: InterfaceMetadata holds information for a single interface.
type InterfaceMetadata struct {
	Name    string   `json:"name"`
	Methods []Method `json:"methods"`
}

// NEW: ImportMetadata holds information for a single import spec.
type ImportMetadata struct {
	Name string `json:"name,omitempty"` // Alias for the import (e.g., "f" in f "fmt")
	Path string `json:"path"`
}

// NEW: ValueSpecMetadata holds information for constants or variables.
type ValueSpecMetadata struct {
	Names []string `json:"names"`           // e.g., ["a", "b"] for `const a, b = 1, 2`
	Type  string   `json:"type,omitempty"`  // The explicit type, if any
	Value string   `json:"value,omitempty"` // The string representation of the value(s)
}

// CodeMetadata is the unified container for all extracted information.
type CodeMetadata struct {
	Type     string            `json:"type"` // "function", "struct", "import", "constant", "variable", "interface"
	FilePath string            `json:"file_path"`
	Function *FunctionMetadata `json:"function,omitempty"`
	Struct   *StructMetadata   `json:"struct,omitempty"`
	// NEW: Added fields for the new types.
	// Import    *ImportMetadata    `json:"import,omitempty"`
	Constant *ValueSpecMetadata `json:"constant,omitempty"`
	// Variable  *ValueSpecMetadata `json:"variable,omitempty"`
	// Interface *InterfaceMetadata `json:"interface,omitempty"`
}

func main() {
	rootDirs := []string{"..", "../internal"} // Adjust as needed

	outputFile, err := os.Create("codebase_map.jsonl")
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer outputFile.Close()

	encoder := json.NewEncoder(outputFile)

	fmt.Println("Starting codebase analysis...")

	for _, root := range rootDirs {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			// Ignore test files and this specific file itself.
			if !info.IsDir() && strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") && !strings.HasSuffix(path, "Map.go") {
				parseFile(path, encoder)
			}
			return nil
		})
		if err != nil {
			fmt.Printf("Error walking directory %s: %v\n", root, err)
		}
	}

	fmt.Println("Analysis complete. Output written to codebase_map.jsonl")
}

// parseFile reads and analyzes a single Go file.
func parseFile(path string, encoder *json.Encoder) {
	fset := token.NewFileSet()
	fileBytes, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("Warning: could not read file %s: %v\n", path, err)
		return
	}

	node, err := parser.ParseFile(fset, path, fileBytes, parser.ParseComments)
	if err != nil {
		fmt.Printf("Warning: could not parse file %s: %v\n", path, err)
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			// Handle function declarations (no changes here)
			if decl.Body == nil {
				return true // Skip function declarations without a body
			}
			funcMeta := FunctionMetadata{
				Name:       decl.Name.Name,
				Parameters: extractFields(decl.Type.Params, fileBytes),
				Returns:    extractFields(decl.Type.Results, fileBytes),
				Body:       string(fileBytes[decl.Body.Lbrace : decl.Body.Rbrace+1]),
			}
			funcMeta.Signature = buildFuncSignature(funcMeta.Name, funcMeta.Parameters, funcMeta.Returns)
			metadata := CodeMetadata{Type: "function", FilePath: path, Function: &funcMeta}
			if err := encoder.Encode(metadata); err != nil {
				fmt.Printf("Warning: failed to encode function metadata for %s: %v\n", funcMeta.Name, err)
			}

		case *ast.GenDecl:
			// NEW: Expanded this section to handle all generic declaration types.
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				// case *ast.ImportSpec:
				// 	path, _ := strconv.Unquote(s.Path.Value)
				// 	importMeta := ImportMetadata{Path: path}
				// 	if s.Name != nil {
				// 		importMeta.Name = s.Name.Name
				// 	}
				// 	metadata := CodeMetadata{Type: "import", FilePath: path, Import: &importMeta}
				// 	if err := encoder.Encode(metadata); err != nil {
				// 		fmt.Printf("Warning: failed to encode import metadata for %s: %v\n", path, err)
				// 	}

				case *ast.ValueSpec: // This handles both `const` and `var`
					valueMeta := extractValueSpec(s, fileBytes)
					var declType string
					if decl.Tok == token.CONST {
						declType = "constant"
						metadata := CodeMetadata{Type: declType, FilePath: path, Constant: &valueMeta}
						if err := encoder.Encode(metadata); err != nil {
							fmt.Printf("Warning: failed to encode constant metadata: %v\n", err)
						}
						// } else if decl.Tok == token.VAR {
						// 	declType = "variable"
						// 	metadata := CodeMetadata{Type: declType, FilePath: path, Variable: &valueMeta}
						// 	if err := encoder.Encode(metadata); err != nil {
						// 		fmt.Printf("Warning: failed to encode variable metadata: %v\n", err)
						// 	}
					}

				case *ast.TypeSpec: // This handles `type` declarations
					switch t := s.Type.(type) {
					case *ast.StructType:
						structMeta := StructMetadata{
							Name:   s.Name.Name,
							Fields: extractFields(t.Fields, fileBytes),
						}
						metadata := CodeMetadata{Type: "struct", FilePath: path, Struct: &structMeta}
						if err := encoder.Encode(metadata); err != nil {
							fmt.Printf("Warning: failed to encode struct metadata for %s: %v\n", structMeta.Name, err)
						}

						// case *ast.InterfaceType:
						// 	interfaceMeta := InterfaceMetadata{
						// 		Name:    s.Name.Name,
						// 		Methods: extractInterfaceMethods(t.Methods, fileBytes),
						// 	}
						// 	metadata := CodeMetadata{Type: "interface", FilePath: path, Interface: &interfaceMeta}
						// 	if err := encoder.Encode(metadata); err != nil {
						// 		fmt.Printf("Warning: failed to encode interface metadata for %s: %v\n", interfaceMeta.Name, err)
						// 	}
					}
				}
			}
		}
		return true
	})
}

// NEW: Helper function to extract const/var details to reduce code duplication.
func extractValueSpec(spec *ast.ValueSpec, source []byte) ValueSpecMetadata {
	meta := ValueSpecMetadata{
		Type: getExprString(spec.Type, source),
	}
	for _, name := range spec.Names {
		meta.Names = append(meta.Names, name.Name)
	}
	var values []string
	for _, val := range spec.Values {
		values = append(values, getExprString(val, source))
	}
	meta.Value = strings.Join(values, ", ")
	return meta
}

// NEW: Helper to extract methods from an interface definition.
// func extractInterfaceMethods(fieldList *ast.FieldList, source []byte) []Method {
// 	if fieldList == nil {
// 		return nil
// 	}
// 	var methods []Method
// 	for _, field := range fieldList.List {
// 		if len(field.Names) > 0 { // It's a method
// 			if funcType, ok := field.Type.(*ast.FuncType); ok {
// 				name := field.Names[0].Name
// 				params := extractFields(funcType.Params, source)
// 				returns := extractFields(funcType.Results, source)
// 				method := Method{
// 					Name:       name,
// 					Parameters: params,
// 					Returns:    returns,
// 					Signature:  buildFuncSignature(name, params, returns),
// 				}
// 				methods = append(methods, method)
// 			}
// 		}
// 		// Note: This does not handle embedded interfaces currently.
// 	}
// 	return methods
// }

// extractFields extracts field data for structs, function params, etc.
func extractFields(fieldList *ast.FieldList, source []byte) []Field {
	if fieldList == nil {
		return nil
	}
	var fields []Field
	for _, field := range fieldList.List {
		typeName := getExprString(field.Type, source)
		tag := ""
		if field.Tag != nil {
			unquotedTag, err := strconv.Unquote(field.Tag.Value)
			if err == nil {
				tag = unquotedTag
			}
		}

		if len(field.Names) > 0 {
			for _, name := range field.Names {
				fields = append(fields, Field{Name: name.Name, Type: typeName, Tag: tag})
			}
		} else {
			fields = append(fields, Field{Name: "", Type: typeName})
		}
	}
	return fields
}

// getExprString converts any AST expression into its string representation.
// Renamed from getTypeString for clarity, as it works on more than just types.
func getExprString(expr ast.Expr, source []byte) string {
	if expr == nil {
		return ""
	}
	// The positions from the AST are 1-based, so we adjust for the 0-based byte slice.
	return string(source[expr.Pos()-1 : expr.End()-1])
}

// buildFuncSignature constructs a readable Go function or method signature.
// Renamed from buildSignature for clarity.
func buildFuncSignature(name string, params, returns []Field) string {
	var paramStrings []string
	for _, p := range params {
		if p.Name != "" {
			paramStrings = append(paramStrings, fmt.Sprintf("%s %s", p.Name, p.Type))
		} else {
			paramStrings = append(paramStrings, p.Type)
		}
	}

	var returnStrings []string
	for _, r := range returns {
		if r.Name != "" {
			returnStrings = append(returnStrings, fmt.Sprintf("%s %s", r.Name, r.Type))
		} else {
			returnStrings = append(returnStrings, r.Type)
		}
	}

	paramStr := strings.Join(paramStrings, ", ")
	returnStr := strings.Join(returnStrings, ", ")

	if len(returns) > 1 {
		returnStr = "(" + returnStr + ")"
	}

	// For methods, the 'func' keyword is often omitted in descriptions, but we'll keep it for consistency.
	return strings.TrimSpace(fmt.Sprintf("func %s(%s) %s", name, paramStr, returnStr))
}
