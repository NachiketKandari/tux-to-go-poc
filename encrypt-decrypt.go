package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Get passkey from environment variable
	passkey := os.Getenv("PASSKEY")
	if passkey == "" {
		log.Fatal("PASSKEY not found in .env file")
	}

	if len(os.Args) < 3 {
		fmt.Println("Usage:")
		fmt.Println("  Encrypt: go run main.go encrypt <input_file> <output_file>")
		fmt.Println("  Decrypt: go run main.go decrypt <input_file> <output_file>")
		os.Exit(1)
	}

	operation := os.Args[1]
	inputFile := os.Args[2]
	outputFile := os.Args[3]

	switch operation {
	case "encrypt":
		encryptFile(inputFile, outputFile, passkey)
		fmt.Printf("Successfully encrypted %s to %s\n", inputFile, outputFile)
	case "decrypt":
		decryptFile(inputFile, outputFile, passkey)
		fmt.Printf("Successfully decrypted %s to %s\n", inputFile, outputFile)
	default:
		log.Fatal("Invalid operation. Use 'encrypt' or 'decrypt'")
	}
}

// deriveKey creates a 32-byte key from passphrase using SHA-256
func deriveKey(passphrase string) []byte {
	hash := sha256.Sum256([]byte(passphrase))
	return hash[:]
}

// encryptFile reads plaintext, encrypts it, and writes base64-encoded ciphertext
func encryptFile(inputFile, outputFile, passkey string) {
	// Read plaintext from file
	plaintext, err := os.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("Failed to read input file: %v", err)
	}

	// Derive 32-byte key from passkey
	key := deriveKey(passkey)

	// Create AES cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		log.Fatalf("Failed to create cipher: %v", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		log.Fatalf("Failed to create GCM: %v", err)
	}

	// Generate random nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		log.Fatalf("Failed to generate nonce: %v", err)
	}

	// Encrypt the plaintext
	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

	// Encode to base64 for text file storage
	encoded := base64.StdEncoding.EncodeToString(ciphertext)

	// Write encrypted data to output file
	err = os.WriteFile(outputFile, []byte(encoded), 0644)
	if err != nil {
		log.Fatalf("Failed to write output file: %v", err)
	}
}

// decryptFile reads base64-encoded ciphertext, decrypts it, and writes plaintext
func decryptFile(inputFile, outputFile, passkey string) {
	// Read base64-encoded ciphertext from file
	encodedData, err := os.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("Failed to read input file: %v", err)
	}

	// Decode from base64
	ciphertext, err := base64.StdEncoding.DecodeString(string(encodedData))
	if err != nil {
		log.Fatalf("Failed to decode base64: %v", err)
	}

	// Derive 32-byte key from passkey
	key := deriveKey(passkey)

	// Create AES cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		log.Fatalf("Failed to create cipher: %v", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		log.Fatalf("Failed to create GCM: %v", err)
	}

	// Extract nonce from ciphertext
	nonceSize := gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		log.Fatal("Ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

	// Decrypt the ciphertext
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		log.Fatalf("Failed to decrypt: %v", err)
	}

	// Write decrypted data to output file
	err = os.WriteFile(outputFile, plaintext, 0644)
	if err != nil {
		log.Fatalf("Failed to write output file: %v", err)
	}
}
