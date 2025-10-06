package main

// import (
// 	"encoding/json"
// 	"fmt"
// 	"go/ast"
// 	"go/parser"
// 	"go/token"
// 	"os"
// 	"path/filepath"
// 	"strconv"
// 	"strings"
// )

// // Field represents a single parameter, return value, or struct field.
// // It now includes a 'Tag' for struct fields.
// type Field struct {
// 	Name string `json:"name"`
// 	Type string `json:"type"`
// 	Tag  string `json:"tag,omitempty"` // omitempty will hide it for function params/returns
// }

// // FunctionMetadata holds the structured information for a single function.
// type FunctionMetadata struct {
// 	Name       string  `json:"name"`
// 	Signature  string  `json:"signature"`
// 	Parameters []Field `json:"parameters"`
// 	Returns    []Field `json:"returns"`
// 	Body       string  `json:"body"`
// }

// // StructMetadata holds the structured information for a single struct.
// type StructMetadata struct {
// 	Name   string  `json:"name"`
// 	Fields []Field `json:"fields"`
// }

// // CodeMetadata is a container to unify the output for functions and structs.
// type CodeMetadata struct {
// 	Type     string            `json:"type"` // "function" or "struct"
// 	FilePath string            `json:"file_path"`
// 	Function *FunctionMetadata `json:"function,omitempty"`
// 	Struct   *StructMetadata   `json:"struct,omitempty"`
// }

// func main() {
// 	// 1. MODIFICATION: Updated to match your project structure.
// 	// We scan the root directory for handlers and main.go, and the internal directory.
// 	rootDirs := []string{"..", "../internal"}

// 	outputFile, err := os.Create("codebase_map.jsonl")
// 	if err != nil {
// 		fmt.Printf("Error creating output file: %v\n", err)
// 		os.Exit(1)
// 	}
// 	defer outputFile.Close()

// 	encoder := json.NewEncoder(outputFile)

// 	fmt.Println("Starting codebase analysis...")

// 	for _, root := range rootDirs {
// 		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
// 			if err != nil {
// 				return err
// 			}

// 			// We only care about non-test Go source files.
// 			// Ignoring this go code.
// 			if !info.IsDir() && strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") && !strings.HasSuffix(path, "Map.go") {
// 				parseFile(path, encoder)
// 			}
// 			return nil
// 		})
// 		if err != nil {
// 			fmt.Printf("Error walking directory %s: %v\n", root, err)
// 		}
// 	}

// 	fmt.Println("Analysis complete. Output written to codebase_map.jsonl")
// }

// // parseFile reads a single Go file, parses its AST, extracts metadata,
// // and writes it to the JSONL output file.
// func parseFile(path string, encoder *json.Encoder) {
// 	fset := token.NewFileSet()
// 	fileBytes, err := os.ReadFile(path)
// 	if err != nil {
// 		fmt.Printf("Warning: could not read file %s: %v\n", path, err)
// 		return
// 	}

// 	node, err := parser.ParseFile(fset, path, fileBytes, parser.ParseComments)
// 	if err != nil {
// 		fmt.Printf("Warning: could not parse file %s: %v\n", path, err)
// 		return
// 	}

// 	ast.Inspect(node, func(n ast.Node) bool {
// 		// 2. MODIFICATION: Use a type switch to handle different declaration types.
// 		switch decl := n.(type) {
// 		case *ast.FuncDecl:
// 			// Handle function declarations
// 			if decl.Body == nil {
// 				return true // Skip function declarations without a body (e.g., in interfaces)
// 			}

// 			funcMeta := FunctionMetadata{
// 				Name:       decl.Name.Name,
// 				Parameters: extractFields(decl.Type.Params, fileBytes),
// 				Returns:    extractFields(decl.Type.Results, fileBytes),
// 				Body:       string(fileBytes[decl.Body.Lbrace : decl.Body.Rbrace+1]),
// 			}
// 			funcMeta.Signature = buildSignature(funcMeta)

// 			metadata := CodeMetadata{
// 				Type:     "function",
// 				FilePath: path,
// 				Function: &funcMeta,
// 			}

// 			if err := encoder.Encode(metadata); err != nil {
// 				fmt.Printf("Warning: failed to encode function metadata for %s in %s: %v\n", funcMeta.Name, path, err)
// 			}

// 		case *ast.GenDecl:
// 			// Handle generic declarations (for types, constants, vars)
// 			if decl.Tok == token.TYPE {
// 				for _, spec := range decl.Specs {
// 					if typeSpec, ok := spec.(*ast.TypeSpec); ok {
// 						if structType, ok := typeSpec.Type.(*ast.StructType); ok {
// 							// We found a struct definition!
// 							structMeta := StructMetadata{
// 								Name:   typeSpec.Name.Name,
// 								Fields: extractFields(structType.Fields, fileBytes),
// 							}

// 							metadata := CodeMetadata{
// 								Type:     "struct",
// 								FilePath: path,
// 								Struct:   &structMeta,
// 							}

// 							if err := encoder.Encode(metadata); err != nil {
// 								fmt.Printf("Warning: failed to encode struct metadata for %s in %s: %v\n", structMeta.Name, path, err)
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 		return true
// 	})
// }

// // extractFields now also handles struct field tags.
// func extractFields(fieldList *ast.FieldList, source []byte) []Field {
// 	if fieldList == nil {
// 		return nil
// 	}
// 	var fields []Field
// 	for _, field := range fieldList.List {
// 		typeName := getTypeString(field.Type, source)
// 		tag := ""
// 		if field.Tag != nil {
// 			// Unquote the tag string to get its raw value
// 			unquotedTag, err := strconv.Unquote(field.Tag.Value)
// 			if err == nil {
// 				tag = unquotedTag
// 			}
// 		}

// 		if len(field.Names) > 0 {
// 			// Named fields (e.g., "ID int `json:\"id\"`")
// 			for _, name := range field.Names {
// 				fields = append(fields, Field{Name: name.Name, Type: typeName, Tag: tag})
// 			}
// 		} else {
// 			// Unnamed fields (e.g., in return types like "(string, error)")
// 			fields = append(fields, Field{Name: "", Type: typeName})
// 		}
// 	}
// 	return fields
// }

// // getTypeString converts an AST expression for a type into a string representation.
// func getTypeString(expr ast.Expr, source []byte) string {
// 	if expr == nil {
// 		return ""
// 	}
// 	return string(source[expr.Pos()-1 : expr.End()-1])
// }

// // buildSignature constructs a readable Go function signature string from metadata.
// func buildSignature(meta FunctionMetadata) string {
// 	var params []string
// 	for _, p := range meta.Parameters {
// 		if p.Name != "" {
// 			params = append(params, fmt.Sprintf("%s %s", p.Name, p.Type))
// 		} else {
// 			params = append(params, p.Type)
// 		}
// 	}

// 	var returns []string
// 	for _, r := range meta.Returns {
// 		if r.Name != "" {
// 			returns = append(returns, fmt.Sprintf("%s %s", r.Name, r.Type))
// 		} else {
// 			returns = append(returns, r.Type)
// 		}
// 	}

// 	paramStr := strings.Join(params, ", ")
// 	returnStr := strings.Join(returns, ", ")

// 	if len(returns) > 1 {
// 		returnStr = "(" + returnStr + ")"
// 	}
// 	// Trim space if there are no return values
// 	return strings.TrimSpace(fmt.Sprintf("func %s(%s) %s", meta.Name, paramStr, returnStr))
// }
