// Example 1: Zero-Cost Abstractions with Iterators vs Manual Loops
// In many languages, using higher-order functions comes with a performance penalty
// In Rust, the iterator version often compiles to the same efficient machine code as manual loops

// Manual loop approach (what you might do in other languages for performance)
fn sum_even_squares_manual(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..numbers.len() {
        let num = numbers[i];
        if num % 2 == 0 {
            sum += num * num;
        }
    }
    sum
}

// Iterator approach (more readable, yet just as fast in Rust)
fn sum_even_squares_iterator(numbers: &[i32]) -> i32 {
    numbers
        .iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum()
}

// Example 2: Memory Management Without Garbage Collection
// Demonstrating how Rust's ownership system eliminates common performance issues

// In many languages: Hidden allocations, copies, and GC pressure
fn process_text_other_lang(text: String) -> String {
    // Implicitly creates multiple copies and temporary allocations
    let lines = text.split('\n'); // Hidden allocation in many languages
    let filtered = lines.filter(|line| !line.trim().is_empty()); // Another allocation
    let processed = filtered.map(|line| line.trim().to_uppercase()); // More allocations
    processed.collect::<Vec<_>>().join("\n") // Final allocation
}

// In Rust: Explicit and efficient with zero hidden costs
fn process_text_rust(text: &str) -> String {
    text.split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_uppercase())
        .collect::<Vec<_>>()
        .join("\n")
}

// Example 3: String Handling - Slices vs Owned Data
// Demonstrating how Rust's type system guides you toward efficient choices

// Inefficient approach - unnecessary ownership transfers and allocations
fn extract_domain_inefficient(email: String) -> String {
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return String::new();
    }
    parts[1].to_string()
}

// Efficient approach - using references to avoid allocations
fn extract_domain_efficient(email: &str) -> &str {
    if let Some(at_pos) = email.find('@') {
        &email[at_pos + 1..]
    } else {
        ""
    }
}

// Example 4: Compile-time Evaluation
// Rust's const generics and const functions enable computation at compile time

// Computing Fibonacci at runtime (like most languages would)
fn fibonacci_runtime(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let mut a = 0;
    let mut b = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

// Computing Fibonacci at compile time, Without const, It’s Always Runtime
const fn fibonacci_compile_time(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => {
            let mut a = 0;
            let mut b = 1;
            let mut i = 2;
            while i <= n {
                let temp = a + b;
                a = b;
                b = temp;
                i += 1;
            }
            b
        }
    }
}

// Using compile-time evaluation
const FIB_10: u64 = fibonacci_compile_time(10); // Computed at compile time!

static FIB_10: u64 = fibonacci_compile_time(10); // Compile-time

let arr: [i32; fibonacci_compile_time(5) as usize] = [0; 5]; // Compile-time

let fib_11 = fibonacci_compile_time(11); // Runtime

// Example 5: Safe Concurrency Without Performance Overhead
// Demonstrating how Rust's ownership rules enable efficient parallelism

use std::thread;

// In many languages: Locks, synchronization, or copying data
fn parallel_sum_other_langs(data: &[i32]) -> i32 {
    // Would typically involve locks, mutexes, or copying data
    // Often leads to contention or excessive memory usage
    // Implementation omitted for brevity
    0 // Placeholder
}

// In Rust: Efficient parallelism with compile-time safety
fn parallel_sum_rust(data: &[i32]) -> i32 {
    if data.len() < 1000 {
        // For small arrays, just do it sequentially
        return data.iter().sum();
    }

    // For larger arrays, split the work
    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);

    // Process each half in parallel with zero-copy slices
    let left_handle = thread::spawn(move || left.iter().sum::<i32>());

    let right_sum = right.iter().sum::<i32>();
    let left_sum = left_handle.join().unwrap();

    left_sum + right_sum
}

// Example 6: Avoiding Dynamic Dispatch with Generics and Traits
// Showing how Rust can eliminate runtime type checking overhead

// In many languages: Runtime polymorphism with performance cost
trait ShapeOtherLang {
    fn area(&self) -> f64;
}

struct CircleOtherLang {
    radius: f64,
}

impl ShapeOtherLang for CircleOtherLang {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// Would incur virtual dispatch overhead in most languages
fn total_area_dynamic(shapes: &[Box<dyn ShapeOtherLang>]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// In Rust: Static dispatch with generics - zero overhead
trait Shape {
    fn area(&self) -> f64;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// No runtime dispatch needed - monomorphization creates optimized code for each type
fn total_area<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Example 7: Move Semantics - Avoiding Hidden Copies
// Demonstrating how Rust's move semantics prevent unexpected performance issues

struct LargeData {
    // Imagine this contains megabytes of data
    payload: Vec<u8>,
}

// In many languages: Implicit, expensive copying
fn process_data_other_lang(data: LargeData) -> LargeData {
    // Many languages would implicitly copy 'data' here
    // This could be very expensive but hidden from the developer
    LargeData {
        payload: data.payload, // More copying!
    }
}

// In Rust: Explicit transfers of ownership
fn process_data_rust(mut data: LargeData) -> LargeData {
    // Ownership is moved, no hidden copies
    // We can modify in place for efficiency
    data.payload.push(42);
    data // Transfer ownership back to caller
}

// Or, even more efficient when appropriate
fn process_data_rust_ref(data: &mut LargeData) {
    // No ownership transfer at all, just borrowed access
    data.payload.push(42);
}

// Example 8: Inlining and Optimizing Small Functions
// Demonstrating how Rust's compiler aggressively optimizes code

// This small function will be inlined automatically
#[inline]
fn add_one(x: i32) -> i32 {
    x + 1
}

fn process_numbers(numbers: &[i32]) -> Vec<i32> {
    // The compiler will inline add_one, eliminating function call overhead
    numbers.iter().map(|&x| add_one(x)).collect()
}

// Example 9: Stack vs Heap Allocation
// Showing how Rust encourages stack allocation for better performance

// In many languages: Hidden heap allocations
fn create_array_other_lang() -> Vec<i32> {
    // Most languages would heap-allocate this array
    vec![1, 2, 3, 4, 5]
}

// In Rust: Explicit choice between stack and heap
fn create_array_stack() -> [i32; 5] {
    // Stack allocated - no allocation overhead, better cache locality
    [1, 2, 3, 4, 5]
}

fn create_array_heap() -> Vec<i32> {
    // Explicitly heap allocated when needed
    vec![1, 2, 3, 4, 5]
}

// Example 10: Compile-time Bounds Checking Elimination
// Showing how Rust's compiler can eliminate unnecessary bounds checks

fn sum_array(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // The compiler can prove this loop is safe and eliminate bounds checks
    for i in 0..arr.len() {
        sum += arr[i]; // No runtime bounds check needed!
    }

    sum
}

// More advanced: Using unsafe when appropriate with clear safety boundaries
fn sum_array_optimized(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // Explicitly avoiding bounds checks when we're certain it's safe
    unsafe {
        for i in 0..arr.len() {
            sum += *arr.get_unchecked(i);
        }
    }

    sum
}

// Example 11: Efficient Error Handling with Result
// Showing how Rust's Result type provides zero-cost error handling

// In many languages: Exceptions with significant overhead
// try {
//     doSomething();
//     doSomethingElse();
// } catch (Exception e) {
//     handleError(e);
// }

// In Rust: Zero-cost abstraction for error handling
fn process_data_result(input: &str) -> Result<i32, &'static str> {
    // The ? operator provides elegant error handling with no performance penalty
    let value = parse_value(input)?;
    let processed = transform_value(value)?;
    Ok(processed)
}

fn parse_value(input: &str) -> Result<i32, &'static str> {
    match input.parse::<i32>() {
        Ok(n) => Ok(n),
        Err(_) => Err("Failed to parse input"),
    }
}

fn transform_value(n: i32) -> Result<i32, &'static str> {
    if n < 0 {
        Err("Value cannot be negative")
    } else {
        Ok(n * 2)
    }
}

// Example 12: Efficient Enums with Pattern Matching
// Demonstrating how Rust's enums avoid the overhead of OOP hierarchies

// In OOP languages: Virtual dispatch and downcasting
// class Shape { public: virtual double area() = 0; };
// class Circle : public Shape { ... };
// class Rectangle : public Shape { ... };
//
// double process(Shape* shape) {
//     if (Circle* c = dynamic_cast<Circle*>(shape)) {
//         // Circle-specific logic
//     } else if (Rectangle* r = dynamic_cast<Rectangle*>(shape)) {
//         // Rectangle-specific logic
//     }
// }

// In Rust: Efficient tagged unions with pattern matching
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            // No runtime type checking or downcasting needed
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }
}

// Example 13: SIMD Vectorization
// Showing how Rust allows explicit SIMD instructions for maximum performance

use std::arch::x86_64::*;

// Safe wrapper around SIMD operations
#[cfg(target_arch = "x86_64")]
fn sum_vector_simd(data: &[f32]) -> f32 {
    let len = data.len();
    let remainder = len % 4;
    let simd_len = len - remainder;

    let mut sum = 0.0;

    // Process 4 elements at a time using SIMD
    if is_x86_feature_detected!("sse2") {
        unsafe {
            let mut sum_vec = _mm_setzero_ps();

            for i in (0..simd_len).step_by(4) {
                let v = _mm_loadu_ps(data[i..].as_ptr());
                sum_vec = _mm_add_ps(sum_vec, v);
            }

            // Extract and sum the 4 float values from the vector
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), sum_vec);
            sum = temp.iter().sum::<f32>();
        }
    }

    // Add the remaining elements
    for i in simd_len..len {
        sum += data[i];
    }

    sum
}

// Example 14: Efficient Binary Serialization with Zero-Copy
// Demonstrating how Rust can parse binary data without unnecessary copies

#[repr(C, packed)]
struct NetworkPacket {
    packet_type: u8,
    flags: u8,
    length: u16,
    data: [u8; 1024],
}

// In many languages: Parse by copying bytes
// function parsePacket(buffer) {
//     const type = buffer.readUInt8(0);
//     const flags = buffer.readUInt8(1);
//     const length = buffer.readUInt16LE(2);
//     const data = Buffer.alloc(length);
//     buffer.copy(data, 0, 4, 4 + length);
//     return { type, flags, length, data };
// }

// In Rust: Zero-copy parsing with memory mapping
fn parse_packet(buffer: &[u8]) -> Option<&NetworkPacket> {
    if buffer.len() < std::mem::size_of::<NetworkPacket>() {
        return None;
    }

    // Safely interpret the buffer as a NetworkPacket without copying
    unsafe {
        let packet = &*(buffer.as_ptr() as *const NetworkPacket);
        Some(packet)
    }
}

// Example 15: Memory Pooling for Frequent Allocations
// Showing explicit memory reuse patterns that other languages handle via GC

struct ObjectPool<T> {
    objects: Vec<Option<T>>,
}

impl<T> ObjectPool<T> {
    fn new(capacity: usize) -> Self {
        let mut objects = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            objects.push(None);
        }
        ObjectPool { objects }
    }

    fn acquire(&mut self, create_fn: impl FnOnce() -> T) -> usize {
        // Find an empty slot or create new
        for (idx, slot) in self.objects.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(create_fn());
                return idx;
            }
        }

        // No empty slots, add a new one
        let idx = self.objects.len();
        self.objects.push(Some(create_fn()));
        idx
    }

    fn release(&mut self, idx: usize) {
        if idx < self.objects.len() {
            self.objects[idx] = None;
        }
    }

    fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.objects.len() {
            return None;
        }
        self.objects[idx].as_ref()
    }
}

// Example 16: Efficient Binary Search with No Dynamic Allocation
// Demonstrating allocation-free algorithms

// Many languages: Hidden allocations in standard library
// function binarySearch(arr, target) {
//     let left = 0;
//     let right = arr.length - 1;
//     while (left <= right) {
//         const mid = Math.floor((left + right) / 2);
//         if (arr[mid] === target) return mid;
//         if (arr[mid] < target) left = mid + 1;
//         else right = mid - 1;
//     }
//     return -1;
// }

// In Rust: Zero allocation binary search
fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    // No allocation, works on any ordered slice
    slice.binary_search(target).ok()
}

// Custom implementation that makes zero allocations
fn binary_search_manual<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = slice.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match slice[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
        }
    }

    None
}

// Example 17: Cache-Friendly Data Structures
// Demonstrating how Rust enables designing for the memory hierarchy

// Structure of Arrays (SoA) vs Array of Structures (AoS)
// SoA is often more cache-friendly for batch processing

// Array of Structures (traditional OOP approach)
struct ParticleAoS {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    velocity_x: f32,
    velocity_y: f32,
    velocity_z: f32,
}

// Structure of Arrays (cache-optimized approach)
struct ParticleSystemSoA {
    position_x: Vec<f32>,
    position_y: Vec<f32>,
    position_z: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    velocity_z: Vec<f32>,
}

impl ParticleSystemSoA {
    fn update_positions(&mut self, dt: f32) {
        // Process each component in a tight loop - better cache utilization
        for i in 0..self.position_x.len() {
            self.position_x[i] += self.velocity_x[i] * dt;
        }

        for i in 0..self.position_y.len() {
            self.position_y[i] += self.velocity_y[i] * dt;
        }

        for i in 0..self.position_z.len() {
            self.position_z[i] += self.velocity_z[i] * dt;
        }
    }
}

// Example 18: Custom Allocators
// Showing how Rust allows complete control over memory allocation strategies

use std::alloc::{GlobalAlloc, Layout, System};

// A custom allocator that tracks allocations
struct TracingAllocator;

unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        // In a real implementation, we'd log this allocation
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        // In a real implementation, we'd log this deallocation
    }
}

// To use: #[global_allocator] static ALLOCATOR: TracingAllocator = TracingAllocator;

// Example 19: Efficient Text Processing with No Allocations
// Demonstrating zero-allocation string manipulation

fn count_words(text: &str) -> usize {
    // No allocations needed - we just iterate through the string once
    text.split_whitespace().count()
}

fn is_palindrome(text: &str) -> bool {
    // Convert to lowercase without allocation by working on UTF-8 bytes
    let bytes: Vec<_> = text
        .bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .map(|b| b.to_ascii_lowercase())
        .collect();

    let len = bytes.len();
    for i in 0..len / 2 {
        if bytes[i] != bytes[len - 1 - i] {
            return false;
        }
    }

    true
}

// Example 20: Memory-Mapped Files for Large Data Processing
// Showing efficient handling of data larger than RAM

use memmap2::MmapOptions;
use std::fs::File;

fn count_lines_in_large_file(path: &str) -> std::io::Result<usize> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Process gigabytes of data without loading it all into memory
    let mut count = 0;
    for byte in &mmap {
        if *byte == b'\n' {
            count += 1;
        }
    }

    Ok(count)
}
